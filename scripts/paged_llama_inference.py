import argparse
import itertools
import os
import time

import torch
import torch._inductor.config
from torch import distributed as dist

from fms.models import get_model
from fms.utils import generation, tokenizers
from fms_extras.modules.speculator import Speculator
from fms_extras.utils.generation import speculative_generate
import fms_extras.models.paged_llama

# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--speculator_path",
    type=str,
    required=True,
    help="Path to the checkpoint containing speculator weights (single .pth file, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)

parser.add_argument(
    "--checkpoint_sharding",
    type=str,
    default=None,
    help="type of weight sharding. E.g. tensor-parallel (tp), None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)

parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument("--context_file", type=str, default=None, help="File to summarize")

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

# torch.set_default_device(device)
torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

model = get_model(
    "paged_llama",
    args.variant,
    model_path=args.model_path,
    checkpoint_sharding=args.checkpoint_sharding,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
    norm_eps=1e-6,
)
decode_model = None

tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading speculator")
speculator = Speculator(model.width, 4096, model.config.src_vocab_size, n_predict=3)
speculator.load_state_dict(
    torch.load(args.speculator_path, map_location=device)["model_state"]
)
speculator = speculator.to(device)
print("loading complete on rank", local_rank)

print("initializing paged cache")
# cache setup
from fms_extras.utils.cache.paged import PagedKVCacheManager

use_cache = True
kv_cache_manager = PagedKVCacheManager(
    model.config.nlayers,
    model.config.nheads,
    model.config.emb_dim,
    kv_heads=model.config.kvheads,
    tensor_parallel_size=dist.get_world_size() if args.distributed else 1,
    dtype=torch.get_default_dtype(),
    device=device,
)
print("cache initialization complete on rank", local_rank)

def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


def pad_prompt(prompt, pad_len, pad_token="<unk>"):
    to_pad = pad_len - len(prompt)
    if to_pad == 0:
        return prompt

    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    pad_ids = [pad_id] * to_pad
    pads = torch.tensor(pad_ids, device=device)
    return torch.cat((pads, prompt))

def print_result(result, inp, n_steps):
    if local_rank != 0:
        return
    # stop at EOS token if present
    result = generation.truncate_after_eos(
        result, tokenizer.convert_tokens_to_ids("</s>")
    )
    # print(result)
    # print(tokenizer.convert_ids_to_tokens(result))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
    print(f"{len(result) - len(inp)} tokens in {n_steps} steps")
    print()


def infer(ids, warmup):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("==================")

    result, n_steps, generated_token_time_out = speculative_generate(
        model,
        ids,
        speculator,
        new_tokens=100,
        max_seq_len=model.config.max_expected_seq_len,
        kv_cache_manager=kv_cache_manager,
        decode_model=decode_model,
    )
    if not warmup:
        total_tokens = 0
        for i in range(len(result)):
            print_result(result[i], ids[i], n_steps)
            total_tokens += (len(result[i]) - len(ids[i]))
        avg_tokens = total_tokens / len(result)
        print(f"time per token: {generated_token_time_out / avg_tokens}")


if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    decode_model = model
    decode_model = torch.compile(decode_model, mode=args.compile_mode, fullgraph=True)
    model = torch.compile(model, fullgraph=True, dynamic=True)
    speculator = torch.compile(speculator, mode=args.compile_mode)

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup."
)
prompt2 = template.format("Explain some popular greetings in Spanish.")
prompt3 = template.format("Explain to me why ignorance is bliss.")
prompt4 = template.format(
    "I have just come into a very large sum of money. I received the money from my parents who told me I could do whatever I want with it. My first thought was to go to a financial advisor. Provide me a list of things that I can do with my new found wealth."
)

prompt1 = ids_for_prompt(prompt1)
prompt2 = ids_for_prompt(prompt2)
prompt3 = ids_for_prompt(prompt3)
prompt4 = ids_for_prompt(prompt4)

# ids = [prompt1, prompt2, prompt3, prompt4]
ids = [prompt1]

infer(ids, warmup=True)
print("generating output", local_rank)
infer(ids, warmup=False)
