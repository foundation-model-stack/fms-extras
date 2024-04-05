import argparse
import itertools
import os
import time

import torch
import torch._inductor.config
from fms.models import get_model
from fms.utils import generation, tokenizers
from torch import distributed as dist

import fms_extras.models.paged_llama
from fms_extras.models.speculator import MLPSpeculator
from fms_extras.utils.generation import paged_generate, speculative_generate


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
# torchrun --nproc_per_node=1 scripts/inference.py --variant=7b --model_path=~/models/7B-F --tokenizer=~/models/tokenizer.model --model_source=meta --speculator_path=~/models/speculator_7B_F.pth --compile

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
    default=None,
    help="Path to the checkpoint containing speculator weights (single .pth file, not HF weights)",
)
parser.add_argument(
    "--speculator_source",
    type=str,
    default="fms",
    choices=["hf", "fms"],
    help="Source format of speculator weights",
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
parser.add_argument(
    "--batch_input",
    action="store_true",
    help="use a batch of prompts as input (note this is still wip for reduce-overhead=True)",
)

args = parser.parse_args()

if args.batch_input and args.compile and args.compile_mode == "reduce-overhead":
    print(
        "setting compile_mode to default as cudagraphs is not yet supported with batches"
    )
    compile_mode = "default"
else:
    compile_mode = args.compile_mode

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

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
speculator = None
if args.speculator_path is not None:
    print("loading speculator")
    if args.speculator_source == "fms":
        speculator = MLPSpeculator(
            model.config.emb_dim, 4096, model.config.src_vocab_size, n_predict=3
        )
        speculator.load_state_dict(
            torch.load(args.speculator_path, map_location=device)["model_state"]
        )
    elif args.speculator_source == "hf":
        from fms_extras.models.hf.modeling_mlp_speculator import (
            MLPSpeculatorPreTrainedModel,
        )

        speculator = MLPSpeculatorPreTrainedModel.from_pretrained(
            args.speculator_path, device_map=device
        )
    else:
        print("speculator format must be one of fms or hf")
        exit()
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

    cudagraphs = compile_mode == "reduce-overhead"

    if speculator:
        result, n_steps, ttft, generated_token_time_out = speculative_generate(
            model,
            ids,
            speculator,
            kv_cache_manager,
            new_tokens=100,
            max_seq_len=model.config.max_expected_seq_len,
            decode_model=decode_model,
            # todo: we can only reduce-overhead for now when batch size is 1
            flattening=not (args.compile and compile_mode == "reduce-overhead"),
            cudagraphs=cudagraphs,
        )
    else:
        result, n_steps, ttft, generated_token_time_out = paged_generate(
            model,
            ids,
            kv_cache_manager,
            max_new_tokens=100,
            max_seq_len=model.config.max_expected_seq_len,
            do_sample=False,
            decode_model=decode_model,
            cudagraphs=cudagraphs,
        )
    if not warmup:
        total_tokens = 0
        for i in range(len(result)):
            print_result(result[i], ids[i], n_steps)
            total_tokens += len(result[i]) - len(ids[i])
        avg_tokens = total_tokens / len(result)
        print(f"time to first token: {ttft}")
        print(f"time per token (decode): {generated_token_time_out / avg_tokens}")


if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    decode_model = model
    decode_model = torch.compile(decode_model, mode=compile_mode, fullgraph=True)
    model = torch.compile(model, fullgraph=True, dynamic=True)
    if speculator:
        speculator = torch.compile(speculator, mode=compile_mode)
        speculator.generate_suffixes = torch.compile(
            speculator.generate_suffixes, mode=compile_mode
        )

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

prompt1 = template.format("Provide a list of instructions for preparing chicken soup.")
prompt2 = template.format("Explain some popular greetings in Spanish.")
prompt3 = template.format("Explain to me why ignorance is bliss.")
prompt4 = template.format(
    "I have just come into a very large sum of money. I received the money from my parents who told me I could do whatever I want with it. My first thought was to go to a financial advisor. Provide me a list of things that I can do with my new found wealth."
)

prompt1 = ids_for_prompt(prompt1)
prompt2 = ids_for_prompt(prompt2)
prompt3 = ids_for_prompt(prompt3)
prompt4 = ids_for_prompt(prompt4)

if args.batch_input:
    ids = [prompt1, prompt2, prompt3, prompt4]
else:
    ids = [prompt1]

infer(ids, warmup=True)
print("generating output", local_rank)
infer(ids, warmup=True)
infer(ids, warmup=False)
