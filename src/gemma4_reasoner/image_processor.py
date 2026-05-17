"""Image preprocessing for Gemma 4's vision encoder."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import requests
from PIL import Image

from .config import ImageTokenBudget

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


class ImageProcessor:
    """Handles image loading, validation, and encoding for Gemma 4.

    Gemma 4's vision encoder supports:
    - Variable aspect ratios
    - Variable resolution (via configurable token budget: 70–1120)
    - Pan & Scan for high-resolution images
    """

    @staticmethod
    def load(source: Union[str, Path, Image.Image]) -> Image.Image:
        """Load an image from a file path, URL, or PIL Image.

        Args:
            source: File path, HTTP(S) URL, or PIL Image.Image

        Returns:
            PIL Image in RGB mode

        Raises:
            ValueError: If the file format is not supported
            requests.HTTPError: If URL fetch fails
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")

        source_str = str(source)

        if source_str.startswith(("http://", "https://")):
            logger.debug(f"Fetching image from URL: {source_str}")
            response = requests.get(source_str, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        return Image.open(path).convert("RGB")

    @staticmethod
    def to_base64(image: Image.Image, fmt: str = "PNG") -> str:
        """Encode a PIL Image as a base64 string."""
        buffer = BytesIO()
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def get_info(image: Image.Image) -> dict:
        """Get metadata about an image."""
        return {
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.width / image.height, 3),
            "mode": image.mode,
            "estimated_tokens": ImageProcessor.estimate_tokens(image),
        }

    @staticmethod
    def estimate_tokens(image: Image.Image) -> int:
        """Estimate vision token count for an image.

        Gemma 4 uses adaptive resolution. At the default 896×896 resolution,
        images encode to ~256 soft tokens. The actual token count depends on
        the configurable token budget (70–1120).
        """
        base_tokens = 256
        base_pixels = 896 * 896
        pixels = image.width * image.height

        if pixels > base_pixels:
            scale = pixels / base_pixels
            return min(int(base_tokens * (1 + (scale - 1) * 0.5)), 1120)
        return base_tokens

    @staticmethod
    def apply_token_budget(
        image: Image.Image,
        budget: ImageTokenBudget,
    ) -> Image.Image:
        """Optionally resize an image to match a token budget.

        For very high resolution images, this can reduce compute by
        downsampling to a resolution appropriate for the budget.
        """
        max_pixels = {
            ImageTokenBudget.MIN: 320 * 320,
            ImageTokenBudget.LOW: 512 * 512,
            ImageTokenBudget.MEDIUM: 768 * 768,
            ImageTokenBudget.HIGH: 1024 * 1024,
            ImageTokenBudget.MAX: 1664 * 1664,
        }

        target = max_pixels[budget]
        current = image.width * image.height

        if current <= target:
            return image

        scale = (target / current) ** 0.5
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        logger.debug(f"Resizing image: {image.width}x{image.height} -> {new_w}x{new_h}")
        return image.resize((new_w, new_h), Image.LANCZOS)
