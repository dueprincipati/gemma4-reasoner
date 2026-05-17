"""Document parsing tool — tables, receipts, forms, handwriting."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..reasoner import Gemma4Reasoner


class DocumentParser:
    """Parse structured data from document images."""

    def __init__(self, reasoner: Gemma4Reasoner):
        self.reasoner = reasoner

    def parse_table(self, image: Union[str, Path]) -> str:
        """Extract table data as markdown."""
        result = self.reasoner.analyze_document(image, doc_type="table")
        return result.response

    def parse_receipt(self, image: Union[str, Path]) -> str:
        """Extract receipt/invoice data."""
        result = self.reasoner.analyze_document(image, doc_type="receipt")
        return result.response

    def parse_form(self, image: Union[str, Path]) -> str:
        """Extract form fields and values."""
        result = self.reasoner.analyze_document(image, doc_type="form")
        return result.response

    def parse_diagram(self, image: Union[str, Path]) -> str:
        """Analyze a diagram or schematic."""
        result = self.reasoner.analyze_document(image, doc_type="diagram")
        return result.response

    def transcribe_handwriting(self, image: Union[str, Path]) -> str:
        """Transcribe handwritten text."""
        result = self.reasoner.analyze_document(image, doc_type="handwriting")
        return result.response
