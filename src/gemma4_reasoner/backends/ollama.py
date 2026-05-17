"""Ollama backend for Gemma 4."""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from ..config import ReasoningConfig
from ..image_processor import ImageProcessor
from ..models import ReasoningResult
from .base import BaseBackend

logger = logging.getLogger(__name__)


class OllamaBackend(BaseBackend):
    """Run Gemma 4 via Ollama.

    Requirements:
        pip install ollama
        ollama pull gemma4:e4b
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        try:
            import ollama
            self._client = ollama
        except ImportError:
            raise ImportError(
                "Ollama backend requires the 'ollama' package. "
                "Install with: pip install ollama"
            )
        self._model_id = config.model_size.ollama_id
        logger.info(f"Ollama backend initialized: {self._model_id}")

    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
    ) -> ReasoningResult:
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_predict": self.config.max_tokens,
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message: dict = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = [
                ImageProcessor.to_base64(img) for img in images
            ]
        messages.append(user_message)

        logger.debug(f"Generating with {self._model_id}, {len(images or [])} images")
        response = self._client.chat(
            model=self._model_id,
            messages=messages,
            options=options,
            stream=False,
        )

        raw = response["message"]["content"]
        final, thinking = self._parse_thinking(raw)

        return ReasoningResult(
            response=final,
            thinking=thinking,
            model=self._model_id,
            num_images=len(images or []),
            raw={"ollama_response": response},
        )
