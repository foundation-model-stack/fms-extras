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
from fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorPreTrainedModel
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
    "--speculator_variant",
    type=str,
    default="840m",
    help="The model variant (configuration) to benchmark. E.g. 840m, 1.4b, 2b, etc.",
)
parser.add_argument(
    "--speculator_source",
    type=str,
    default=None,
    choices=["hf"],
    help="Source format of speculator weights. Note: If the weights path specified in speculator_path are not local and "
    "the source is hf, the weights will be pulled using the normal Huggingface from_pretrained method.",
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
# top_k_tokens_per_head
parser.add_argument(
    "--top_k_tokens_per_head",
    type=lambda s: list(map(int, s.split(","))),
    default=[5, 3, 2],
    help="Number of tokens to consider from each head when forming the candidate tree. For each candidate branch in the tree, head n produces topk[n] additional sub-branches.",
)
parser.add_argument(
    "--prompt_type",
    type=str,
    choices=["chat", "code"],
    default="chat",
    help="type of prompts to be used, either chat or code",
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
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

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
)
decode_model = None

tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
speculator = None
if args.speculator_path is not None:
    print("loading speculator")
    # todo: handling of remote weights in get_model
    is_local = os.path.exists(args.speculator_path) or args.speculator_source != "hf"
    if is_local:
        speculator = get_model(
            "mlp_speculator",
            f"llama.{args.variant}.{args.speculator_variant}",
            model_path=args.speculator_path,
            source=args.speculator_source,
            device_type=args.device_type,
        )
    else:
        from fms_extras.models.hf.modeling_mlp_speculator import (
            MLPSpeculatorPreTrainedModel,
        )

        speculator = MLPSpeculatorPreTrainedModel.from_pretrained(
            args.speculator_path, device_map=args.device_type
        ).speculator
    speculator = speculator.to(device)
    if len(args.top_k_tokens_per_head) != speculator.n_predict:
        print(
            "length of top_k_tokens_per_head must be equal to the speculator's number of heads (n_predict)"
        )
        exit()
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
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = [tokenizer.bos_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


def print_result(result, inp, n_steps):
    if local_rank != 0:
        return
    # stop at EOS token if present
    result = generation.truncate_after_eos(result, tokenizer.eos_token_id)
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
            threshes=args.top_k_tokens_per_head,
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

if args.prompt_type == "chat":
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

    prompt1 = template.format(
        "Provide a list of instructions for preparing chicken soup."
    )
    prompt2 = template.format("Explain some popular greetings in Spanish.")
    prompt3 = template.format("Explain to me why ignorance is bliss.")
    prompt4 = template.format(
        "I have just come into a very large sum of money. I received the money from my parents who told me I could do whatever I want with it. My first thought was to go to a financial advisor. Provide me a list of things that I can do with my new found wealth."
    )

elif args.prompt_type == "code":
    template = "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{}\n[/INST]"
    prompt1 = template.format("Write a bubble sort function in python.")
    prompt2 = template.format(
        "Using the Java streams API, write a simple function which will get the cumulative sum of a list of integers."
    )
    prompt3 = template.format(
        "In bash, how do I list all directories and sub-directories which contain a .py file."
    )
    prompt4 = template.format(
        "Write a simple decorator in python which will modify all string inputs to ints if possible."
    )

else:
    print("prompt_type must be one of chat or code")
    exit()


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
