"""Tests for PromptBuilder."""

from __future__ import annotations

from gemma4_reasoner.prompt_builder import PromptBuilder


class TestPromptBuilder:
    def test_system_prompt_thinking(self):
        prompt = PromptBuilder.system_prompt(thinking=True)
        assert "<|think|>" in prompt

    def test_system_prompt_standard(self):
        prompt = PromptBuilder.system_prompt(thinking=False)
        assert "<|think|>" not in prompt

    def test_build_basic(self):
        prompt = PromptBuilder.build("What is this?")
        assert "What is this?" in prompt

    def test_build_with_context(self):
        prompt = PromptBuilder.build("What is this?", context="This is a chart")
        assert "This is a chart" in prompt
        assert "What is this?" in prompt

    def test_chart_analysis(self):
        prompt = PromptBuilder.chart_analysis()
        assert "chart" in prompt.lower()
        assert "axes" in prompt.lower()

    def test_table_extraction(self):
        prompt = PromptBuilder.table_extraction()
        assert "table" in prompt.lower()

    def test_receipt_parsing(self):
        prompt = PromptBuilder.receipt_parsing()
        assert "receipt" in prompt.lower() or "merchant" in prompt.lower()

    def test_math_solve(self):
        prompt = PromptBuilder.math_solve()
        assert "step-by-step" in prompt.lower() or "step" in prompt.lower()

    def test_math_solve_with_extra(self):
        prompt = PromptBuilder.math_solve(extra="Use algebra")
        assert "algebra" in prompt.lower()

    def test_screen_describe(self):
        prompt = PromptBuilder.screen_describe()
        assert "screen" in prompt.lower() or "interface" in prompt.lower()

    def test_screen_interact(self):
        prompt = PromptBuilder.screen_interact()
        assert "interactive" in prompt.lower()

    def test_screen_accessibility(self):
        prompt = PromptBuilder.screen_accessibility()
        assert "accessibility" in prompt.lower() or "wcag" in prompt.lower()
