"""Base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

from ..config import ReasoningConfig
from ..models import ReasoningResult


class BaseBackend(ABC):
    """Abstract base for Gemma 4 inference backends."""

    def __init__(self, config: ReasoningConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
    ) -> ReasoningResult:
        """Generate a response from Gemma 4.

        Args:
            prompt: The user text prompt
            images: Optional list of PIL Images
            system_prompt: Optional system prompt override

        Returns:
            ReasoningResult with response and optional thinking content
        """
        ...

    def _parse_thinking(self, raw: str) -> tuple[str, Optional[str]]:
        """Parse thinking content from Gemma 4 output.

        Gemma 4 thinking format:
        <|channel>thought\n...reasoning...\n<|channel>analysis\n...answer...

        Returns:
            (final_response, thinking_content_or_None)
        """
        if "<|channel>thought" not in raw:
            return raw, None

        # Split on the analysis channel marker
        parts = raw.split("<|channel>analysis>")
        if len(parts) != 2:
            return raw, None

        thought_section = parts[0]
        analysis_section = parts[1].strip()

        # Clean up the thought section
        thinking = thought_section.replace("<|channel>thought", "").strip()

        return analysis_section, thinking if thinking else None
