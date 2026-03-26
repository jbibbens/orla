"""Convenience constructors for :class:`~pyorla.types.LLMBackend`."""

from __future__ import annotations

import random
import string

from pyorla.types import LLMBackend

_BACKEND_TYPE_OPENAI = "openai"
_BACKEND_TYPE_SGLANG = "sglang"


def _random_backend_name() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=4)) + "-" + "".join(
        random.choices(string.ascii_lowercase, k=4)
    )


def _model_id_for_backend_type(backend_type: str, model_id: str) -> str:
    return f"{backend_type}:{model_id}"


def new_vllm_backend(model_id: str, endpoint: str) -> LLMBackend:
    """Create a vLLM backend (OpenAI-compatible API)."""
    return LLMBackend(
        name=_random_backend_name(),
        endpoint=endpoint,
        type=_BACKEND_TYPE_OPENAI,
        model_id=_model_id_for_backend_type(_BACKEND_TYPE_OPENAI, model_id),
    )


def new_sglang_backend(model_id: str, endpoint: str) -> LLMBackend:
    """Create an SGLang backend."""
    return LLMBackend(
        name=_random_backend_name(),
        endpoint=endpoint,
        type=_BACKEND_TYPE_SGLANG,
        model_id=_model_id_for_backend_type(_BACKEND_TYPE_OPENAI, model_id),
    )


def new_ollama_backend(model_id: str, endpoint: str) -> LLMBackend:
    """Create an Ollama backend.

    endpoint should be the base Ollama URL (e.g. "http://ollama:11434");
    "/v1" is appended automatically.
    """
    return LLMBackend(
        name=_random_backend_name(),
        endpoint=endpoint.rstrip("/") + "/v1",
        type=_BACKEND_TYPE_OPENAI,
        model_id=_model_id_for_backend_type(_BACKEND_TYPE_OPENAI, model_id),
    )
