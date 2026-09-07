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

import torch
import uvloop
from benchmark_dataset import (AIMODataset, BurstGPTDataset,
                               ConversationDataset, InstructCoderDataset,
                               RandomDataset, SampleRequest, ShareGPTDataset,
                               SonnetDataset, VisionArenaDataset, SpecBenchDataset, UltraChatDataset)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from spexis.engine.arg_utils import AsyncEngineArgs, EngineArgs
from spexis.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from spexis.inputs import TextPrompt, TokensPrompt
from spexis.lora.request import LoRARequest
from spexis.outputs import RequestOutput
from spexis.sampling_params import BeamSearchParams
from spexis.utils import FlexibleArgumentParser, merge_async_iterators

def run_vllm(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> tuple[float, Optional[list[RequestOutput]]]:
    from spexis import LLM, SamplingParams
    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(
        llm.llm_engine.model_config.max_model_len >= (
            request.prompt_len + request.expected_output_len)
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
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            ))
    lora_requests: Optional[list[LoRARequest]] = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]

    use_beam_search = False

    outputs = None
    if not use_beam_search:
        start = time.perf_counter()
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests,
                               use_tqdm=True)
        end = time.perf_counter()
    else:
        assert lora_requests is None, "BeamSearch API does not support LoRA"
        prompts = [request.prompt for request in requests]
        # output_len should be the same for all requests.
        output_len = requests[0][2]
        for request in requests:
            assert request.expected_output_len == output_len
        start = time.perf_counter()
        llm.beam_search(
            prompts,
            BeamSearchParams(
                beam_width=n,
                max_tokens=output_len,
                ignore_eos=True,
            ))
        end = time.perf_counter()
    return end - start, outputs


def run_vllm_chat(
        requests: list[SampleRequest],
        n: int,
        engine_args: EngineArgs,
        disable_detokenize: bool = False) -> tuple[float, list[RequestOutput]]:
    """
    Run vLLM chat benchmark. This function is recommended ONLY for benchmarking
    multimodal models as it properly handles multimodal inputs and chat
    formatting. For non-multimodal models, use run_vllm() instead.
    """
    from spexis import LLM, SamplingParams
    llm = LLM(**dataclasses.asdict(engine_args))

    assert all(
        llm.llm_engine.model_config.max_model_len >= (
            request.prompt_len + request.expected_output_len)
        for request in requests), (
            "Please ensure that max_model_len is greater than the sum of "
            "prompt_len and expected_output_len for all requests.")

    prompts = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            ))
    start = time.perf_counter()
    outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start, outputs


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
        lora_requests: list[Optional[LoRARequest]] = []
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
            lora_requests.append(request.lora_request)

        generators = []
        
        # torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(msg="async_generate")
        start = time.perf_counter()
        for i, (prompt, sp,
                lr) in enumerate(zip(prompts, sampling_params, lora_requests)):
            generator = llm.generate(prompt,
                                     sp,
                                     lora_request=lr,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        
        async for i, res in all_gens:
            pass

        end = time.perf_counter()

        torch.cuda.nvtx.range_pop()
        # torch.cuda.cudart().cudaProfilerStop()
        
        return end - start

async def run_vllm_async_multiturn(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    tokenizer: PreTrainedTokenizerBase,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
    temperature: float = 0.0,
    force_generation_step: Optional[bool] = None,
    generation_step: Optional[int] = None,
) -> tuple[float, int, int, int]:
    """
    Run specbench multiturn with batched inference per turn.
    Returns (elapsed_time, total_inferences, total_prompt_tokens, total_output_tokens).
    """
    from spexis import SamplingParams

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        assert all(
            llm.model_config.max_model_len >= (request.prompt_len +
                                               request.expected_output_len)
            for request in requests), (
                "Please ensure that max_model_len is greater than the sum of"
                " prompt_len and expected_output_len for all requests.")

        # Add the requests to the engine. Use request_id "test0", "test1", ... for conversations and turn_results.
        request_id_fmt = "test{}"
        conversations: dict[str, list[dict[str, str]]] = {}
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_inferences = 0
        for idx, request in enumerate(requests):
            request_id = request_id_fmt.format(idx)
            conversation = [{"role": "system", "content": "You are a helpful assistant."}]
            conversation.extend([
                {"role": "user", "content": request.turns[0]},
            ])
            conversations[request_id] = conversation

        start = time.perf_counter()
        for turns in range(len(requests[0].turns)):
            print(f"turns: {turns}")
            prompts: list[Union[TextPrompt, TokensPrompt]] = []
            sampling_params: list[SamplingParams] = []
            lora_requests: list[Optional[LoRARequest]] = []
            for idx, request in enumerate(requests):
                request_id = request_id_fmt.format(idx)
                conversation = conversations[request_id]
                try:
                    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except Exception as e:
                    print(f"Error applying chat template: {e}")
                    continue
                prompt_ids = tokenizer(prompt).input_ids
                prompts.append(TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=request.multi_modal_data))
                sampling_params.append(
                    SamplingParams(
                        n=n,
                        temperature=temperature,
                        top_p=1.0,
                        ignore_eos=False,
                        max_tokens=llm.model_config.max_model_len,
                        detokenize=not disable_detokenize,
                    ))
                lora_requests.append(request.lora_request)

            generators = []
    
            # torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(msg="async_generate")

            for i, (prompt, sp,
                    lr) in enumerate(zip(prompts, sampling_params, lora_requests)):
                generator = llm.generate(prompt,
                                         sp,
                                         lora_request=lr,
                                         request_id=f"test{i}")
                generators.append(generator)
            all_gens = merge_async_iterators(*generators)
            # Collect each request's output by request_id (same key as conversations)
            turn_results: dict[str, Any] = {}
            async for i, res in all_gens:
                if res.finished:
                    turn_results[res.request_id] = [res.prompt, res.outputs[0].text, res.prompt_token_ids, res.outputs[0].token_ids]
                pass

            # Accumulate token counts from this turn
            for request_id, out in turn_results.items():
                prompt_token_ids = out[2]
                output_token_ids = out[3]
                total_prompt_tokens += len(prompt_token_ids) if prompt_token_ids else 0
                total_output_tokens += len(output_token_ids) if output_token_ids else 0
                total_inferences += 1

            # Update conversations for next turn (if not last turn)
            if turns < len(requests[0].turns) - 1:
                for request_id, out in turn_results.items():
                    idx = int(request_id.replace("test", ""))
                    text = out[1]
                    conversations[request_id].append({"role": "assistant", "content": text})
                    conversations[request_id].append({"role": "user", "content": requests[idx].turns[turns + 1]})

        end = time.perf_counter()

        torch.cuda.nvtx.range_pop()
        # torch.cuda.cudart().cudaProfilerStop()
        
        return end - start, total_inferences, total_prompt_tokens, total_output_tokens

async def _generate_one_and_get_text(
    llm,
    prompt: str,
    sampling_params,
    lora_request,
    request_id: str,
    tokenizer: PreTrainedTokenizerBase,
    disable_detokenize: bool,
) -> tuple[str, int, int]:
    """Run one generate, consume the async stream, return (decoded_text, prompt_len, output_len)."""
    prompt_ids = tokenizer(prompt).input_ids
    prompt_len = len(prompt_ids)
    text_prompt = TextPrompt(prompt=prompt)
    gen = llm.generate(text_prompt, sampling_params, lora_request=lora_request, request_id=request_id)
    last_out = None
    async for out in gen:
        last_out = out
    if not last_out or not getattr(last_out, "outputs", None):
        return "", prompt_len, 0
    out0 = last_out.outputs[0] if last_out.outputs else None
    if not out0:
        return "", prompt_len, 0
    token_ids = getattr(out0, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(out0, "sampled_token_ids", None)
    if isinstance(token_ids, list):
        output_len = len(token_ids)
        text = tokenizer.decode(token_ids, skip_special_tokens=True) if not disable_detokenize else getattr(out0, "text", "") or tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        output_len = len(token_ids) if token_ids is not None else 0
        text = getattr(out0, "text", "") or (tokenizer.decode(list(token_ids), skip_special_tokens=True) if token_ids is not None else "")

    return text, prompt_len, output_len


async def run_vllm_multiturn_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    tokenizer: PreTrainedTokenizerBase,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
    temperature: float = 0.0,
    force_generation_step: Optional[bool] = None,
    generation_step: Optional[int] = None,
) -> tuple[float, int, int, int]:
    """
    Run specbench multiturn: for each request with .turns, run inference per turn,
    appending model response and next user turn to the prompt until turns are exhausted.
    Returns (elapsed_time, total_inferences, total_prompt_tokens, total_output_tokens).
    """
    from spexis import SamplingParams

    multireqs = [r for r in requests if getattr(r, "turns", None) and len(r.turns) > 0]
    if not multireqs:
        return 0.0, 0, 0, 0

    max_output_len = multireqs[0].expected_output_len
    if force_generation_step and generation_step is not None:
        max_output_len = generation_step

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            ignore_eos=False,
            max_tokens=llm.model_config.max_model_len,
            detokenize=not disable_detokenize,
        )
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_inferences = 0
        start = time.perf_counter()
        for req_idx, req in enumerate(multireqs):
            turns = req.turns
            conversation = [{"role": "system", "content": "You are a helpful assistant."}]
            conversation.append({"role": "user", "content": turns[0]})
            for turn_idx in range(len(turns)):
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                rid = f"multiturn_{req_idx}_{turn_idx}"
                text, plen, olen = await _generate_one_and_get_text(
                    llm, prompt, sampling_params, req.lora_request, rid, tokenizer, disable_detokenize
                )
                total_prompt_tokens += plen
                total_output_tokens += olen
                total_inferences += 1
                if turn_idx < len(turns) - 1:
                    conversation.append({"role": "assistant", "content": text})
                    conversation.append({"role": "user", "content": turns[turn_idx + 1]})

        end = time.perf_counter()
    return end - start, total_inferences, total_prompt_tokens, total_output_tokens


def run_hf(
    requests: list[SampleRequest],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
    disable_detokenize: bool = False,
) -> float:
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: list[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        if not disable_detokenize:
            # Include the decoding time.
            tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: list[SampleRequest],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [request.prompt for request in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
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
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if args.backend == "vllm-chat":
            sample_kwargs["enable_multimodal_chat"] = True
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset.")
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs['dataset_split'] = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs['dataset_subset'] = args.hf_subset
            common_kwargs['dataset_split'] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
    elif args.dataset_name == "specbench":
        # raise NotImplementedError("Spec-Bench dataset is not implemented yet.")
        dataset_cls = SpecBenchDataset
        sample_kwargs["output_len"] = 2048
        sample_kwargs["task"] = args.specbench_task
        # Options for task:
        # "total", "writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem",
        # "humanities", "translation", "summarization", "qa", "math_reasoning", "rag"
    elif args.dataset_name == "ultrachat":
        dataset_cls = UltraChatDataset
        sample_kwargs["output_len"] = 2048 #yjh:todo hard coded
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
    is_multi_modal = any(request.multi_modal_data is not None
                         for request in requests)
    request_outputs: Optional[list[RequestOutput]] = None
    multiturn_metrics: Optional[tuple[int, int, int]] = None  # (num_inferences, prompt_tokens, output_tokens)
    if args.backend == "vllm":
        if args.async_engine:
            if args.dataset_name == "specbench" and getattr(args, "specbench_task", None) == "multiturn":
                elapsed_time, num_inferences, ptok, otok = uvloop.run(
                    run_vllm_async_multiturn(
                        requests,
                        args.n,
                        AsyncEngineArgs.from_cli_args(args),
                        tokenizer,
                        args.disable_frontend_multiprocessing,
                        args.disable_detokenize,
                        args.temperature,
                        args.force_generation_step,
                        args.generation_step,
                    ))
                multiturn_metrics = (num_inferences, ptok, otok)
            else:
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
        else:
            elapsed_time, request_outputs = run_vllm(
                requests, args.n, EngineArgs.from_cli_args(args),
                args.disable_detokenize)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.hf_max_batch_size, args.trust_remote_code,
                              args.disable_detokenize)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    elif args.backend == "vllm-chat":
        elapsed_time, request_outputs = run_vllm_chat(
            requests, args.n, EngineArgs.from_cli_args(args),
            args.disable_detokenize)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if multiturn_metrics:
        num_requests_metric, total_prompt_tokens, total_output_tokens = multiturn_metrics
        total_num_tokens = total_prompt_tokens + total_output_tokens
    elif request_outputs:
        # Note: with the vllm and vllm-chat backends,
        # we have request_outputs, which we use to count tokens.
        total_prompt_tokens = 0
        total_output_tokens = 0
        for ro in request_outputs:
            if not isinstance(ro, RequestOutput):
                continue
            total_prompt_tokens += len(
                ro.prompt_token_ids) if ro.prompt_token_ids else 0
            total_output_tokens += sum(
                len(o.token_ids) for o in ro.outputs if o)
        total_num_tokens = total_prompt_tokens + total_output_tokens
        num_requests_metric = len(requests)
    else:
        total_num_tokens = sum(r.prompt_len + r.expected_output_len
                               for r in requests)
        total_output_tokens = sum(r.expected_output_len for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens
        num_requests_metric = len(requests)

    if is_multi_modal and args.backend != "vllm-chat":
        print("\033[91mWARNING\033[0m: Multi-modal request with "
              f"{args.backend} backend detected. The "
              "following metrics are not accurate because image tokens are not"
              " counted. See vllm-project/vllm/issues/9778 for details.")
        # TODO(vllm-project/vllm/issues/9778): Count multi-modal token length.
        # vllm-chat backend counts the image tokens now
    
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

    # === Backend Validation ===
    valid_backends = {"vllm", "hf", "mii", "vllm-chat"}
    if args.backend not in valid_backends:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # === Dataset Configuration ===
    if not args.dataset and not args.dataset_path:
        print(
            "When dataset path is not set, it will default to random dataset")
        args.dataset_name = 'random'
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
            getattr(args, "hf_subset", None) is not None
            or getattr(args, "hf_split", None) is not None):
        warnings.warn("--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
                      stacklevel=2)
    elif args.dataset_name == "hf":
        if args.dataset_path in (
                VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
                | ConversationDataset.SUPPORTED_DATASET_PATHS):
            assert args.backend == "vllm-chat", f"{args.dataset_path} needs to use vllm-chat as the backend."  #noqa: E501
        elif args.dataset_path in (InstructCoderDataset.SUPPORTED_DATASET_PATHS
                                   | AIMODataset.SUPPORTED_DATASET_PATHS):
            assert args.backend == "vllm", f"{args.dataset_path} needs to use vllm as the backend."  #noqa: E501
        else:
            raise ValueError(
                f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != 'random' and args.random_range_ratio is not None:
        warnings.warn("--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
                      stacklevel=2)

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if args.dataset_name not in {"random", "sonnet", None
                                 } and args.prefix_len is not None:
        warnings.warn("--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
                      stacklevel=2)

    # === LoRA Settings ===
    if getattr(args, "enable_lora", False) and args.backend != "vllm":
        raise ValueError(
            "LoRA benchmarking is only supported for vLLM backend")
    if getattr(args, "enable_lora", False) and args.lora_path is None:
        raise ValueError("LoRA path must be provided when enable_lora is True")

    # === Backend-specific Validations ===
    if args.backend == "hf" and args.hf_max_batch_size is None:
        raise ValueError("HF max batch size is required for HF backend")
    if args.backend != "hf" and args.hf_max_batch_size is not None:
        raise ValueError("HF max batch size is only for HF backend.")

    if args.backend in {"hf", "mii"} and getattr(args, "quantization",
                                                 None) is not None:
        raise ValueError("Quantization is only for vLLM backend.")

    if args.backend == "mii" and args.dtype != "auto":
        raise ValueError("dtype must be auto for MII backend.")
    if args.backend == "mii" and args.n != 1:
        raise ValueError("n must be 1 for MII backend.")
    if args.backend == "mii" and args.tokenizer != args.model:
        raise ValueError(
            "Tokenizer must be the same as the model for MII backend.")
    
    if args.temperature is None:
        raise ValueError("temperature must be given")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii", "vllm-chat"],
                        default="vllm")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "random", "sonnet", "burstgpt", "hf", "specbench", "ultrachat"],
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
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
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
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize the response (i.e. do not include "
              "detokenization time in the measurement)"))
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the lora adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.")
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=None,
        help=f"Number of prefix tokens to be used in RandomDataset "
        "and SonnetDataset. For RandomDataset, the total input "
        "length is the sum of prefix-len (default: "
        f"{RandomDataset.DEFAULT_PREFIX_LEN}) and a random context length "
        "sampled from [input_len * (1 - range_ratio), "
        "input_len * (1 + range_ratio)]. For SonnetDataset, "
        f"prefix_len (default: {SonnetDataset.DEFAULT_PREFIX_LEN}) "
        "controls how much of the input is fixed lines versus "
        "random lines, but the total input length remains approximately "
        "input_len tokens.")
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=None,
        help=f"Range ratio (default : {RandomDataset.DEFAULT_RANGE_RATIO}) "
        "for sampling input/output length, "
        "used only for RandomDataset. Must be in the range [0, 1) to "
        "define a symmetric sampling range "
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )

    # hf dtaset
    parser.add_argument("--hf-subset",
                        type=str,
                        default=None,
                        help="Subset of the HF dataset.")
    parser.add_argument("--hf-split",
                        type=str,
                        default=None,
                        help="Split of the HF dataset.")
    
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
