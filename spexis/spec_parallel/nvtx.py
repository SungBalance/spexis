# SPDX-License-Identifier: Apache-2.0
"""NVTX range helpers that no-op unless profiling is enabled.

The Spexis hot paths carry NVTX ranges for Nsight profiling. They are
off by default and enabled per process (engine core and each worker)
from ``SpexisConfig.enable_nvtx_profile`` via :func:`enable`, so a
normal run pays no NVTX cost.
"""
import torch

_enabled = False


def enable() -> None:
    global _enabled
    _enabled = True


def range_push(msg: str) -> None:
    if _enabled:
        torch.cuda.nvtx.range_push(msg=msg)


def range_pop() -> None:
    if _enabled:
        torch.cuda.nvtx.range_pop()


def mark(msg: str) -> None:
    if _enabled:
        torch.cuda.nvtx.mark(msg=msg)
