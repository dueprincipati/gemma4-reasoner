"""OpenAI-compatible API backend (vLLM, TGI, etc.)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

from ..config import ReasoningConfig
from ..image_processor import ImageProcessor
from ..models import ReasoningResult
from .base import BaseBackend

logger = logging.getLogger(__name__)


class OpenAICompatBackend(BaseBackend):
    """Run Gemma 4 via an OpenAI-compatible API server.

    Works with vLLM, TGI, LiteLLM, etc.

    Set OPENAI_BASE_URL and OPENAI_API_KEY environment variables.
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI-compat backend requires the 'openai' package. "
                "Install with: pip install openai"
            )

        base_url = config.base_url
        api_key = os.getenv("OPENAI_API_KEY", "dummy")

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_id = config.model_size.model_id
        logger.info(f"OpenAI-compat backend initialized: {base_url}")

    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
    ) -> ReasoningResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                b64 = ImageProcessor.to_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )

        raw = response.choices[0].message.content or ""
        final, thinking = self._parse_thinking(raw)

        return ReasoningResult(
            response=final,
            thinking=thinking,
            model=self._model_id,
            num_images=len(images or []),
        )
