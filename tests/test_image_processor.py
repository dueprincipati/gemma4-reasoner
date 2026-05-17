"""Tests for ImageProcessor."""

from __future__ import annotations

import pytest
from PIL import Image

from gemma4_reasoner.image_processor import ImageProcessor
from gemma4_reasoner.config import ImageTokenBudget


class TestImageProcessor:
    def test_load_pil_image(self, sample_image):
        result = ImageProcessor.load(sample_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_load_from_file(self, sample_image_path):
        result = ImageProcessor.load(sample_image_path)
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)

    def test_load_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            ImageProcessor.load(str(path))

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ImageProcessor.load("/nonexistent/image.png")

    def test_to_base64(self, sample_image):
        b64 = ImageProcessor.to_base64(sample_image)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_get_info(self, sample_image):
        info = ImageProcessor.get_info(sample_image)
        assert info["width"] == 224
        assert info["height"] == 224
        assert info["mode"] == "RGB"
        assert "estimated_tokens" in info

    def test_estimate_tokens_small(self):
        img = Image.new("RGB", (224, 224))
        tokens = ImageProcessor.estimate_tokens(img)
        assert tokens <= 256

    def test_estimate_tokens_large(self):
        img = Image.new("RGB", (2000, 2000))
        tokens = ImageProcessor.estimate_tokens(img)
        assert tokens > 256
        assert tokens <= 1120

    def test_apply_token_budget_no_resize(self):
        img = Image.new("RGB", (224, 224))
        result = ImageProcessor.apply_token_budget(img, ImageTokenBudget.MAX)
        assert result.size == (224, 224)

    def test_apply_token_budget_resize(self):
        img = Image.new("RGB", (2000, 2000))
        result = ImageProcessor.apply_token_budget(img, ImageTokenBudget.MIN)
        assert result.size[0] < 2000
        assert result.size[1] < 2000
