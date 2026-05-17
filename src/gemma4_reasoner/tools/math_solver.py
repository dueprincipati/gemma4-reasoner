"""Visual math problem solver."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..reasoner import Gemma4Reasoner


class MathSolver:
    """Solve math problems presented as images."""

    def __init__(self, reasoner: Gemma4Reasoner):
        self.reasoner = reasoner

    def solve(
        self,
        image: Union[str, Path],
        show_work: bool = True,
    ):
        """Solve a visual math problem.

        Args:
            image: Image of the math problem
            show_work: If True (default), uses thinking mode for step-by-step work

        Returns:
            ReasoningResult with .response (answer) and .thinking (work shown)
        """
        return self.reasoner.solve_math(image)

    def solve_with_hint(
        self,
        image: Union[str, Path],
        hint: str,
    ):
        """Solve with a hint or additional instruction."""
        return self.reasoner.solve_math(image, extra=hint)
