"""Prompt construction for Gemma 4.

Follows the official Gemma 4 best practices:
- Thinking mode: enabled via <|think|> token in system prompt
- Image ordering: place images BEFORE text for optimal performance
- Multi-turn: do NOT include thinking content in conversation history
- Sampling: temperature=1.0, top_p=0.95, top_k=64
"""

from __future__ import annotations

from typing import Optional


class PromptBuilder:
    """Builds properly formatted prompts for Gemma 4.

    Gemma 4 uses standard system/user/assistant roles with special
    control tokens for thinking mode:
    - <|think|> in system prompt enables chain-of-thought
    - Model outputs: <|channel>thought\n...reasoning...\n<|channel>analysis\n...answer...
    """

    SYSTEM_THINKING = """<|think|>
You are an expert multimodal reasoning engine powered by Gemma 4.
When presented with visual materials and questions:
1. Carefully observe and describe what you see
2. Identify key elements, relationships, and patterns
3. Apply domain-specific knowledge to interpret the data
4. Reason step-by-step through the problem
5. Synthesize observations into a clear, precise answer"""

    SYSTEM_STANDARD = """You are an expert multimodal reasoning engine powered by Gemma 4.
Analyze the provided visual materials and answer accurately and concisely."""

    @classmethod
    def system_prompt(cls, thinking: bool = True) -> str:
        """Get the system prompt."""
        return cls.SYSTEM_THINKING if thinking else cls.SYSTEM_STANDARD

    @classmethod
    def build(
        cls,
        question: str,
        thinking: bool = True,
        context: Optional[str] = None,
    ) -> str:
        """Build a complete prompt string."""
        parts = []
        if context:
            parts.append(f"Background Context:\n{context}")
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)

    @classmethod
    def chart_analysis(cls) -> str:
        """Optimized prompt for chart/graph analysis."""
        return """Analyze this chart or graph thoroughly. Provide:
1. Chart type (bar, line, pie, scatter, etc.)
2. Axes labels, units, and scales
3. Title and legend information
4. Key data points with specific numerical values
5. Trends, patterns, and correlations
6. Notable outliers or anomalies
7. Overall conclusion or insight the chart conveys"""

    @classmethod
    def table_extraction(cls) -> str:
        """Optimized prompt for table data extraction."""
        return """Extract all data from this table. Provide:
1. Complete data as a markdown table
2. All column headers and row labels
3. Every cell value exactly as shown
4. Any footnotes or special formatting notes
5. The table's title or caption if present"""

    @classmethod
    def receipt_parsing(cls) -> str:
        """Optimized prompt for receipt/invoice parsing."""
        return """Extract all information from this receipt or invoice:
1. Merchant/store name and address
2. Date and time of transaction
3. All items: description, quantity, unit price, total price
4. Subtotal, tax amount, and grand total
5. Payment method
6. Any discounts, promotions, or loyalty info"""

    @classmethod
    def diagram_analysis(cls) -> str:
        """Optimized prompt for diagram/schematic analysis."""
        return """Analyze this diagram thoroughly:
1. Type of diagram (flowchart, circuit, architecture, etc.)
2. All components and their labels
3. Connections and relationships between components
4. Flow directions or signal paths
5. The overall system or process being described
6. How the components interact"""

    @classmethod
    def form_extraction(cls) -> str:
        """Optimized prompt for form field extraction."""
        return """Extract all fields and values from this form:
1. Form type and title
2. All labeled fields and their filled-in values
3. Checkboxes: checked vs. unchecked
4. Signatures and dates
5. Any handwritten annotations"""

    @classmethod
    def handwriting_transcription(cls) -> str:
        """Optimized prompt for handwriting recognition."""
        return """Transcribe all handwritten text in this image:
1. Preserve original formatting and line breaks
2. Maintain the original structure (paragraphs, lists, etc.)
3. Mark unclear words with [unclear]
4. Note any drawings or diagrams present"""

    @classmethod
    def math_solve(cls, extra: Optional[str] = None) -> str:
        """Optimized prompt for visual math problem solving."""
        base = """Solve the mathematical problem shown in the image:
1. Write out the complete problem statement
2. Show step-by-step reasoning
3. Clearly state each mathematical operation
4. Provide the final answer with appropriate units
5. Verify your answer if possible"""
        if extra:
            base += f"\n\nAdditional instructions: {extra}"
        return base

    @classmethod
    def screen_describe(cls) -> str:
        """Prompt for UI/screen description."""
        return """Describe this screen or interface in detail:
1. Application or website name
2. All visible UI elements (buttons, menus, text fields, icons)
3. Current state of the interface
4. All visible text content
5. Layout and navigation structure"""

    @classmethod
    def screen_interact(cls) -> str:
        """Prompt for UI interaction analysis."""
        return """Analyze this screen for user interaction:
1. List all interactive elements with labels
2. Approximate position of each element
3. What actions can the user take?
4. What is the most likely next action?
5. Any disabled or hidden elements?"""

    @classmethod
    def screen_accessibility(cls) -> str:
        """Prompt for accessibility evaluation."""
        return """Evaluate this screen for accessibility (WCAG):
1. Text contrast ratios
2. Alt text presence for images
3. Keyboard navigation support
4. Screen reader compatibility
5. Touch target sizes
6. Specific recommendations for improvement"""
