"""Shared test fixtures."""

from __future__ import annotations

import pytest
from PIL import Image

from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize, BackendType


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a small test image."""
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    return img


@pytest.fixture
def sample_image_path(tmp_path, sample_image) -> str:
    """Save test image to a temp file."""
    path = tmp_path / "test_image.png"
    sample_image.save(path)
    return str(path)


@pytest.fixture
def config() -> ReasoningConfig:
    """Default test config (doesn't require a running backend)."""
    return ReasoningConfig(
        model_size=ModelSize.E4B,
        backend=BackendType.OLLAMA,
        thinking=True,
    )


@pytest.fixture
def reasoner(config) -> Gemma4Reasoner:
    """Create a reasoner instance (backend not connected until used)."""
    return Gemma4Reasoner(config)
