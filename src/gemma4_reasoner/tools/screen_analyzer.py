"""Screen/UI analysis tool for agentic workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..reasoner import Gemma4Reasoner


class ScreenAnalyzer:
    """Analyze screens and user interfaces."""

    def __init__(self, reasoner: Gemma4Reasoner):
        self.reasoner = reasoner

    def describe(self, image: Union[str, Path]) -> str:
        """Describe a screen in detail."""
        result = self.reasoner.analyze_screen(image, task="describe")
        return result.response

    def get_interactable_elements(self, image: Union[str, Path]) -> str:
        """List all interactive elements on a screen."""
        result = self.reasoner.analyze_screen(image, task="interact")
        return result.response

    def check_accessibility(self, image: Union[str, Path]) -> str:
        """Evaluate a screen for accessibility compliance."""
        result = self.reasoner.analyze_screen(image, task="accessibility")
        return result.response
