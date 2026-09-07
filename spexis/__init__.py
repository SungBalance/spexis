# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import spexis.env_override  # isort:skip  # noqa: F401

from spexis.engine.arg_utils import AsyncEngineArgs, EngineArgs
from spexis.engine.async_llm_engine import AsyncLLMEngine
from spexis.engine.llm_engine import LLMEngine
from spexis.entrypoints.llm import LLM
from spexis.executor.ray_utils import initialize_ray_cluster
from spexis.inputs import PromptType, TextPrompt, TokensPrompt
from spexis.model_executor.models import ModelRegistry
from spexis.outputs import (ClassificationOutput, ClassificationRequestOutput,
                          CompletionOutput, EmbeddingOutput,
                          EmbeddingRequestOutput, PoolingOutput,
                          PoolingRequestOutput, RequestOutput, ScoringOutput,
                          ScoringRequestOutput)
from spexis.pooling_params import PoolingParams
from spexis.sampling_params import SamplingParams

__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
