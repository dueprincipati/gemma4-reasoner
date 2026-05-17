"""Core reasoning engine — the main entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .backends import (
    BaseBackend,
    HuggingFaceBackend,
    OllamaBackend,
    OpenAICompatBackend,
)
from .config import BackendType, ImageTokenBudget, ModelSize, ReasoningConfig
from .image_processor import ImageProcessor
from .models import ChatMessage, ReasoningResult
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class Gemma4Reasoner:
    """Multimodal reasoning engine powered by Gemma 4.

    Supports all four model sizes (E2B through 31B), multiple backends,
    thinking mode, document analysis, math solving, and screen understanding.

    Example:
        >>> config = ReasoningConfig(model_size=ModelSize.E4B, backend=BackendType.OLLAMA)
        >>> reasoner = Gemma4Reasoner(config)
        >>> result = reasoner.analyze_document("chart.png", doc_type="chart")
        >>> print(result.response)
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.image_processor = ImageProcessor()
        self.prompt_builder = PromptBuilder()
        self.backend = self._create_backend()
        self._history: list[ChatMessage] = []
        logger.info(
            f"Gemma4Reasoner initialized: model={self.config.model_size.value}, "
            f"backend={self.config.backend.value}, thinking={self.config.thinking}"
        )

    def _create_backend(self) -> BaseBackend:
        """Instantiate the appropriate backend adapter."""
        mapping = {
            BackendType.OLLAMA: OllamaBackend,
            BackendType.HUGGINGFACE: HuggingFaceBackend,
            BackendType.OPENAI_COMPAT: OpenAICompatBackend,
        }
        backend_cls = mapping.get(self.config.backend)
        if not backend_cls:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        return backend_cls(self.config)

    def _load_images(
        self,
        images: list[Union[str, Path, Image.Image]],
    ) -> list[Image.Image]:
        """Load and optionally resize images per the token budget."""
        loaded = [self.image_processor.load(img) for img in images]
        return [
            self.image_processor.apply_token_budget(img, self.config.image_token_budget)
            for img in loaded
        ]

    def _system_prompt(self) -> str:
        return self.prompt_builder.system_prompt(thinking=self.config.thinking)

    # ── Public API ───────────────────────────────────────────────────────

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        question: str = "Describe this image in detail.",
        context: Optional[str] = None,
        thinking: Optional[bool] = None,
    ) -> ReasoningResult:
        """Analyze a single image with a question.

        Args:
            image: File path, URL, or PIL Image
            question: What to ask about the image
            context: Optional background context
            thinking: Override config's thinking mode for this call

        Returns:
            ReasoningResult with response and optional chain-of-thought
        """
        img = self.image_processor.load(image)
        img = self.image_processor.apply_token_budget(img, self.config.image_token_budget)
        img_info = self.image_processor.get_info(img)

        use_thinking = thinking if thinking is not None else self.config.thinking
        prompt = self.prompt_builder.build(
            question=question,
            thinking=use_thinking,
            context=context,
        )

        logger.info(f"Analyzing image {img_info['width']}x{img_info['height']}: {question[:80]}...")
        result = self.backend.generate(
            prompt=prompt,
            images=[img],
            system_prompt=self._system_prompt(),
        )
        result.image_info = img_info
        return result

    def compare_images(
        self,
        images: list[Union[str, Path, Image.Image]],
        question: str = "Compare these images and highlight the key differences.",
        context: Optional[str] = None,
    ) -> ReasoningResult:
        """Compare multiple images.

        Gemma 4 supports interleaved multimodal input — images can be
        freely mixed with text in any order.
        """
        loaded = self._load_images(images)
        total_tokens = sum(
            self.image_processor.estimate_tokens(img) for img in loaded
        )

        prompt = self.prompt_builder.build(
            question=question,
            thinking=self.config.thinking,
            context=context,
        )

        logger.info(f"Comparing {len(loaded)} images (~{total_tokens} vision tokens)")
        result = self.backend.generate(
            prompt=prompt,
            images=loaded,
            system_prompt=self._system_prompt(),
        )
        result.num_images = len(loaded)
        result.total_vision_tokens = total_tokens
        return result

    def analyze_document(
        self,
        image: Union[str, Path, Image.Image],
        doc_type: str = "general",
        query: Optional[str] = None,
    ) -> ReasoningResult:
        """Specialized document analysis with optimized prompts.

        Uses HIGH token budget for OCR and document parsing.

        Args:
            image: Document image
            doc_type: One of: chart, table, receipt, diagram, form, handwriting, general
            query: Custom query (only used when doc_type='general')
        """
        doc_prompts = {
            "chart": self.prompt_builder.chart_analysis(),
            "table": self.prompt_builder.table_extraction(),
            "receipt": self.prompt_builder.receipt_parsing(),
            "diagram": self.prompt_builder.diagram_analysis(),
            "form": self.prompt_builder.form_extraction(),
            "handwriting": self.prompt_builder.handwriting_transcription(),
            "general": query or "Provide a comprehensive analysis of this document.",
        }

        if doc_type not in doc_prompts:
            raise ValueError(
                f"Unknown doc_type '{doc_type}'. Choose from: {list(doc_prompts.keys())}"
            )

        # Use HIGH budget for documents
        original_budget = self.config.image_token_budget
        self.config.image_token_budget = ImageTokenBudget.HIGH

        result = self.analyze_image(image, question=doc_prompts[doc_type])
        result.doc_type = doc_type

        self.config.image_token_budget = original_budget
        return result

    def solve_math(
        self,
        image: Union[str, Path, Image.Image],
        extra: Optional[str] = None,
    ) -> ReasoningResult:
        """Solve a visual math problem with step-by-step reasoning.

        Leverages Gemma 4's strong math performance:
        - 31B: 89.2% on AIME 2026
        - E4B: 42.5% on AIME 2026
        """
        question = self.prompt_builder.math_solve(extra=extra)
        return self.analyze_image(image, question=question, thinking=True)

    def analyze_screen(
        self,
        image: Union[str, Path, Image.Image],
        task: str = "describe",
    ) -> ReasoningResult:
        """UI/Screen understanding for agentic workflows.

        Args:
            image: Screenshot image
            task: One of: describe, interact, accessibility
        """
        task_prompts = {
            "describe": self.prompt_builder.screen_describe(),
            "interact": self.prompt_builder.screen_interact(),
            "accessibility": self.prompt_builder.screen_accessibility(),
        }

        if task not in task_prompts:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(task_prompts.keys())}"
            )

        return self.analyze_image(image, question=task_prompts[task])

    def chat(
        self,
        message: str,
        images: Optional[list[Union[str, Path, Image.Image]]] = None,
    ) -> ReasoningResult:
        """Multi-turn conversation with optional image context.

        Per Gemma 4 best practices, thinking content from previous turns
        is NOT included in the conversation history.
        """
        loaded = self._load_images(images) if images else None

        result = self.backend.generate(
            prompt=message,
            images=loaded,
            system_prompt=self._system_prompt(),
        )

        # Store in history (response only, not thinking)
        self._history.append(ChatMessage(role="user", content=message))
        self._history.append(ChatMessage(role="assistant", content=result.response))

        return result

    def reset_chat(self):
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[ChatMessage]:
        """Get conversation history."""
        return list(self._history)
