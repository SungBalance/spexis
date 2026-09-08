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
"""
import argparse

import torch
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embed_tokens from a target model checkpoint.")
    parser.add_argument("--model", required=True,
                        help="Path to the target model checkpoint directory")
    parser.add_argument("--output", default="embed_tokens.bin",
                        help="Output file (default: ./embed_tokens.bin)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16,
                                                 device_map="cpu")
    embed = model.state_dict()["model.embed_tokens.weight"]
    torch.save({"embed_tokens.weight": embed}, args.output)
    print(f"Saved embed_tokens {tuple(embed.shape)} ({embed.dtype}) "
          f"to {args.output}")


if __name__ == "__main__":
    main()
