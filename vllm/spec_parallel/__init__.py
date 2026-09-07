# SPDX-License-Identifier: Apache-2.0
"""Spexis: speculative pipeline parallelism on top of the vLLM V1 engine.

Modules:
    scheduler                   scheduler that overlaps speculation with
                                normal pipeline execution
    kv_cache_manager            KV cache manager that keeps spec tokens out
                                of the prefix cache
    exitlayer / exit_models     per-stage early-exit (draft) models
    gpu_model_runner            model runner that executes the target model
                                and the exit layer together
    handle_inputbatch           CPU-side embedding batch bookkeeping
    shm                         engine-core <-> worker shared-memory channel
    bool_token                  Ray-DAG sentinel payload
    iteration_log               per-iteration metrics log
    sched_output / stop_check   scheduler output types and stop checks

The lookahead-scheduling half of Spexis (generation-length prediction and
peak-KV projection) lives in :mod:`vllm.lookahead_scheduling`.
"""
