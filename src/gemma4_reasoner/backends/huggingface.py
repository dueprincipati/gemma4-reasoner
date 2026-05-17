"""HuggingFace Transformers backend for Gemma 4."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

from ..config import ReasoningConfig
from ..models import ReasoningResult
from .base import BaseBackend

logger = logging.getLogger(__name__)


class HuggingFaceBackend(BaseBackend):
    """Run Gemma 4 via HuggingFace Transformers.

    Requirements:
        pip install torch transformers accelerate
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_id = self.config.model_size.model_id
        logger.info(f"Loading model: {model_id}")

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info(f"Model loaded on {self._model.device}")

    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
    ) -> ReasoningResult:
        # Build messages — images BEFORE text per Gemma 4 best practices
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        user_content = []
        if images:
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
            )

        generated = outputs[0][input_len:]
        raw = self._processor.decode(generated, skip_special_tokens=False)

        final, thinking = self._parse_thinking(raw)

        return ReasoningResult(
            response=final,
            thinking=thinking,
            model=self.config.model_size.model_id,
            num_images=len(images or []),
        )
