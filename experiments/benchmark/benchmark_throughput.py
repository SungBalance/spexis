# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, Optional, Union

import uvloop
from benchmark_dataset import (SampleRequest, ShareGPTDataset,
                               SpecBenchDataset)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from spexis.engine.arg_utils import AsyncEngineArgs, EngineArgs
from spexis.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from spexis.inputs import TextPrompt, TokensPrompt
from spexis.utils import FlexibleArgumentParser, merge_async_iterators

async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
    temperature: float = 0.0,
    force_generation_step: Optional[bool] = None,
    generation_step: Optional[int] = None,
) -> float:
    from spexis import SamplingParams

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        assert all(
            llm.model_config.max_model_len >= (request.prompt_len +
                                               request.expected_output_len)
            for request in requests), (
                "Please ensure that max_model_len is greater than the sum of"
                " prompt_len and expected_output_len for all requests.")

        # Add the requests to the engine.
        prompts: list[Union[TextPrompt, TokensPrompt]] = []
        sampling_params: list[SamplingParams] = []
        for request in requests:
            prompts.append(
                TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"],
                        multi_modal_data=request.multi_modal_data)
                if "prompt_token_ids" in request.prompt else \
                TextPrompt(prompt=request.prompt,
                           multi_modal_data=request.multi_modal_data))
            if not force_generation_step:
                expected_max = request.expected_output_len
            else:
                expected_max = generation_step

            request.expected_output_len = expected_max

            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=llm.model_config.max_model_len,
                    detokenize=not disable_detokenize,
                ))

        generators = []

        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt,
                                     sp,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        
        async for i, res in all_gens:
            pass

        end = time.perf_counter()

        return end - start

def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k]
            for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "num_requests": args.num_prompts,
        "output_len": args.output_len,
    }

    if args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
    elif args.dataset_name == "specbench":
        dataset_cls = SpecBenchDataset
        sample_kwargs["output_len"] = 2048
        sample_kwargs["task"] = args.specbench_task
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    return dataset_cls(**common_kwargs).sample(**sample_kwargs)


def main(args: argparse.Namespace):
    if args.seed is None:
        args.seed = 0
    print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = get_requests(args, tokenizer)
    elapsed_time = uvloop.run(
        run_vllm_async(
            requests,
            args.n,
            AsyncEngineArgs.from_cli_args(args),
            args.disable_frontend_multiprocessing,
            args.disable_detokenize,
            args.temperature,
            args.force_generation_step,
            args.generation_step,
        ))

    total_num_tokens = sum(r.prompt_len + r.expected_output_len
                           for r in requests)
    total_output_tokens = sum(r.expected_output_len for r in requests)
    total_prompt_tokens = total_num_tokens - total_output_tokens
    num_requests_metric = len(requests)

    print(f"Latency: {elapsed_time:.4f} s")
    print(f"Throughput: {num_requests_metric / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
          f"{total_output_tokens / elapsed_time:.2f} output tokens/s")
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": num_requests_metric,
            "total_num_tokens": total_num_tokens,
            "requests_per_second": num_requests_metric / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


def validate_args(args):
    """
    Validate command-line arguments.
    """

    # === Deprecation and Defaulting ===
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next release. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2)
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model


    if args.temperature is None:
        raise ValueError("temperature must be given")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm"],
                        default="vllm")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "specbench"],
        help="Name of the dataset to benchmark on.",
        default="sharegpt")
    parser.add_argument('--specbench-task',
                        type=str,
                        default='total',
                        )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in\
            the next release. The dataset is expected to "
        "be a json in form of list[dict[..., conversations: "
        "list[dict[..., value: <prompt_or_response>]]]]")
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize the response (i.e. do not include "
              "detokenization time in the measurement)"))
    parser.add_argument('--temperature',
                        type=float,
                        )
    parser.add_argument('--force-generation-step',
                        type=bool,
                        default=False,
                        )
    parser.add_argument('--generation-step',
                        type=int,
                        default=None,
                        )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    validate_args(args)
    main(args)
