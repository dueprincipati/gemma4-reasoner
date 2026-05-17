"""Configuration for the Gemma 4 Reasoner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelSize(Enum):
    """Gemma 4 model variants.

    Architecture details (from the official model card):
    - E2B:  2.3B effective params (5.1B with PLE embeddings), 35 layers, 128K context
    - E4B:  4.5B effective params (8B with PLE embeddings), 42 layers, 128K context
    - 26B:  25.2B total / 3.8B active (MoE, 8 of 128 experts), 30 layers, 256K context
    - 31B:  30.7B dense params, 60 layers, 256K context
    """

    E2B = "e2b"
    E4B = "e4b"
    MOE_26B = "26b"
    DENSE_31B = "31b"

    @property
    def model_id(self) -> str:
        """HuggingFace model identifier."""
        mapping = {
            ModelSize.E2B: "google/gemma-4-e2b-it",
            ModelSize.E4B: "google/gemma-4-e4b-it",
            ModelSize.MOE_26B: "google/gemma-4-26b-a4b-it",
            ModelSize.DENSE_31B: "google/gemma-4-31b-it",
        }
        return mapping[self]

    @property
    def ollama_id(self) -> str:
        """Ollama model identifier."""
        mapping = {
            ModelSize.E2B: "gemma4:e2b",
            ModelSize.E4B: "gemma4:e4b",
            ModelSize.MOE_26B: "gemma4:26b",
            ModelSize.DENSE_31B: "gemma4:31b",
        }
        return mapping[self]

    @property
    def context_length(self) -> int:
        """Maximum context window in tokens."""
        if self in (ModelSize.E2B, ModelSize.E4B):
            return 131_072  # 128K
        return 262_144  # 256K

    @property
    def supports_audio(self) -> bool:
        """Whether the model supports audio input."""
        return self in (ModelSize.E2B, ModelSize.E4B)

    @property
    def memory_requirements_gb(self) -> dict[str, float]:
        """Approximate VRAM requirements by precision (inference only)."""
        requirements = {
            ModelSize.E2B: {"bf16": 9.6, "sfp8": 4.6, "q4_0": 3.2},
            ModelSize.E4B: {"bf16": 15.0, "sfp8": 7.5, "q4_0": 5.0},
            ModelSize.MOE_26B: {"bf16": 48.0, "sfp8": 25.0, "q4_0": 15.6},
            ModelSize.DENSE_31B: {"bf16": 58.3, "sfp8": 30.4, "q4_0": 17.4},
        }
        return requirements[self]


class ImageTokenBudget(Enum):
    """Configurable visual token budgets for Gemma 4's vision encoder.

    Gemma 4 supports variable image resolution through a configurable visual
    token budget. Higher budgets preserve more visual detail at the cost of
    additional compute.
    """

    MIN = 70        # Fast classification, video frames
    LOW = 140       # Captioning
    MEDIUM = 280    # General understanding
    HIGH = 560      # OCR, document parsing
    MAX = 1120      # Fine-grained reading, small text


class BackendType(Enum):
    """Supported inference backends."""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPAT = "openai_compat"


@dataclass
class ReasoningConfig:
    """Complete configuration for the Gemma 4 Reasoner.

    All fields can be set via constructor args or environment variables
    (see from_env()).
    """

    # Model
    model_size: ModelSize = ModelSize.E4B

    # Backend
    backend: BackendType = BackendType.OLLAMA
    base_url: str = "http://localhost:11434"

    # Sampling (Gemma 4 recommended defaults: temp=1.0, top_p=0.95, top_k=64)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 8192

    # Reasoning
    thinking: bool = True

    # Vision
    image_token_budget: ImageTokenBudget = ImageTokenBudget.HIGH

    # Context
    max_context_length: Optional[int] = None  # None = use model default

    @classmethod
    def from_env(cls) -> ReasoningConfig:
        """Build configuration from environment variables."""
        def _enum(env_var: str, enum_cls, default):
            val = os.getenv(env_var)
            if val is None:
                return default
            try:
                return enum_cls(val.lower())
            except ValueError:
                valid = [e.value for e in enum_cls]
                raise ValueError(
                    f"Invalid {env_var}={val}. Must be one of: {valid}"
                )

        def _float(env_var: str, default: float) -> float:
            val = os.getenv(env_var)
            return float(val) if val else default

        def _int(env_var: str, default: int) -> int:
            val = os.getenv(env_var)
            return int(val) if val else default

        def _bool(env_var: str, default: bool) -> bool:
            val = os.getenv(env_var)
            if val is None:
                return default
            return val.lower() in ("1", "true", "yes", "on")

        model_size = _enum("MODEL_SIZE", ModelSize, ModelSize.E4B)
        max_ctx = _int("MAX_CONTEXT_LENGTH", 0)

        return cls(
            model_size=model_size,
            backend=_enum("BACKEND", BackendType, BackendType.OLLAMA),
            base_url=os.getenv("OLLAMA_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:11434")),
            temperature=_float("TEMPERATURE", 1.0),
            top_p=_float("TOP_P", 0.95),
            top_k=_int("TOP_K", 64),
            max_tokens=_int("MAX_TOKENS", 8192),
            thinking=_bool("THINKING", True),
            image_token_budget=_enum("IMAGE_TOKEN_BUDGET", ImageTokenBudget, ImageTokenBudget.HIGH),
            max_context_length=max_ctx if max_ctx > 0 else None,
        )

    def effective_context_length(self) -> int:
        """Get the effective context length."""
        if self.max_context_length:
            return min(self.max_context_length, self.model_size.context_length)
        return self.model_size.context_length
