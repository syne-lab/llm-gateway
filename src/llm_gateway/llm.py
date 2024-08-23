from __future__ import annotations
import os
from llm_gateway.llms import LLMs
from llm_gateway.anthropic_llm import AnthropicLLM
from llm_gateway.llamacpp_llm import LlamaCppLLM
from llm_gateway.transformer_llm import TransformersLLM
from llm_gateway.openai_llm import OpenAILLM

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .llmabc import LLMAbstractBaseClass


def token_from_env(env_var: str, tokens: Dict[str, str]):
    if tokens.get(env_var) is None:
        tokens[env_var] = os.getenv(env_var) if os.getenv(env_var) else None


def LLM(model: LLMs, tokens: Optional[Dict[str, str]] = None) -> LLMAbstractBaseClass:
    """
    Create an LLM instance from the model name. Download and initialize the model if applicable.

    Argument:
        model: The model name.

    """
    if tokens is None:
        tokens = dict()
    for env_var in [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
    ]:
        token_from_env(env_var, tokens)
    engine = model.getInfo().engine
    _model_runner_map = {
        "llama.cpp": LlamaCppLLM,
        "transformers": TransformersLLM,
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
    }
    return _model_runner_map[engine](model, tokens)


def set_model_cache_dir(cache_dir_path: str):
    """Set the cache directory for models."""
    import os

    os.environ["TRANSFORMERS_CACHE"] = cache_dir_path
