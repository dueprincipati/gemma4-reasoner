"""Chart and graph analysis tool."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..reasoner import Gemma4Reasoner


class ChartAnalyzer:
    """Specialized tool for analyzing charts and graphs."""

    def __init__(self, reasoner: Gemma4Reasoner):
        self.reasoner = reasoner

    def analyze(
        self,
        image: Union[str, Path],
        question: str | None = None,
    ):
        """Analyze a chart. Optionally ask a specific question about it."""
        if question:
            return self.reasoner.analyze_image(
                image,
                question=question,
            )
        return self.reasoner.analyze_document(image, doc_type="chart")

    def extract_data(self, image: Union[str, Path]) -> str:
        """Extract all data points from a chart as structured text."""
        result = self.reasoner.analyze_image(
            image,
            question="Extract all data from this chart. Provide the data in a structured format (CSV or markdown table). Include all axis labels, data series names, and exact numerical values.",
        )
        return result.response

    def summarize_trend(self, image: Union[str, Path]) -> str:
        """Summarize the main trend shown in a chart."""
        result = self.reasoner.analyze_image(
            image,
            question="What is the main trend or insight shown in this chart? Summarize in 2-3 sentences.",
        )
        return result.response
