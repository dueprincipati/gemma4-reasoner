"""Backend adapters for different inference engines."""

from .base import BaseBackend
from .ollama import OllamaBackend
from .huggingface import HuggingFaceBackend
from .openai_compat import OpenAICompatBackend

__all__ = [
    "BaseBackend",
    "OllamaBackend",
    "HuggingFaceBackend",
    "OpenAICompatBackend",
]
