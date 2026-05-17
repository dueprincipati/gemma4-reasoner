"""Data models for the Gemma 4 Reasoner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReasoningResult:
    """Result from a reasoning operation."""

    response: str
    thinking: Optional[str] = None
    model: str = ""
    image_info: Optional[dict] = None
    num_images: int = 0
    total_vision_tokens: int = 0
    doc_type: Optional[str] = None
    raw: Optional[dict] = None

    @property
    def has_thinking(self) -> bool:
        return self.thinking is not None and len(self.thinking.strip()) > 0

    def __str__(self) -> str:
        return self.response


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    images: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d
