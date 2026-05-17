"""Gemma 4 Multimodal Reasoner — A production-ready reasoning engine."""

from .config import ReasoningConfig, ModelSize, ImageTokenBudget
from .reasoner import Gemma4Reasoner
from .image_processor import ImageProcessor
from .prompt_builder import PromptBuilder

__version__ = "0.1.0"
__all__ = [
    "Gemma4Reasoner",
    "ReasoningConfig",
    "ModelSize",
    "ImageTokenBudget",
    "ImageProcessor",
    "PromptBuilder",
]
