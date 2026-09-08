# SPDX-License-Identifier: Apache-2.0
"""Extract the target model's embedding table for the Spexis CPU lookup.

With Spexis enabled, the first pipeline stage looks token embeddings up
on the CPU instead of holding a GPU copy of the table, and expects them
in a standalone file (``--cpu-embedding-path``). That file is just the
target model's own ``embed_tokens`` weights, so it is not distributed
with the exit-model artifacts -- generate it yourself from the target
checkpoint you already downloaded:

    python tools/extract_embed_tokens.py \\
        --model /path/to/Llama-3.3-70B-Instruct \\
        --output /path/to/exitmodels/Llama-3.3-70B-Instruct/embed_tokens.bin

Only the shard holding the embedding tensor is read, so this needs a
few GB of host memory rather than the whole checkpoint.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

WEIGHT_NAME = "model.embed_tokens.weight"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embed_tokens from a target model checkpoint.")
    parser.add_argument("--model", required=True,
                        help="Path to the target model checkpoint directory")
    parser.add_argument("--output", default="embed_tokens.bin",
                        help="Output file (default: ./embed_tokens.bin)")
    return parser.parse_args()


def find_shard(model_dir: str) -> str:
    """Return the safetensors file that holds the embedding tensor."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        if WEIGHT_NAME not in weight_map:
            raise KeyError(
                f"{WEIGHT_NAME} not listed in {index_path}")
        return os.path.join(model_dir, weight_map[WEIGHT_NAME])

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return single
    raise FileNotFoundError(
        f"No safetensors checkpoint found under {model_dir}")


def main():
    args = parse_args()

    shard = find_shard(args.model)
    with safe_open(shard, framework="pt", device="cpu") as f:
        embed = f.get_tensor(WEIGHT_NAME)

    # Match the historical format: fp16, keyed "embed_tokens.weight".
    embed = embed.to(torch.float16)
    torch.save({"embed_tokens.weight": embed}, args.output)
    print(f"Saved embed_tokens {tuple(embed.shape)} ({embed.dtype}) "
          f"from {os.path.basename(shard)} to {args.output}")


if __name__ == "__main__":
    main()
