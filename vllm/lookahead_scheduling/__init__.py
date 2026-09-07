# SPDX-License-Identifier: Apache-2.0
"""Spexis lookahead scheduling: generation-length prediction and peak-KV
projection used to admit prefills without triggering future preemptions.

Modules:
    length_predictor            remaining-length prediction heads
                                (spexis_ver >= 2.0)
    length_calc                 expected-generation-length / peak-KV math

The speculative-parallel half of Spexis lives in :mod:`vllm.spec_parallel`.
"""
