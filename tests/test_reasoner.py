"""Tests for Gemma4Reasoner (unit tests that don't require a live backend)."""

from __future__ import annotations

import pytest
from PIL import Image

from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize, BackendType
from gemma4_reasoner.models import ReasoningResult


class TestReasoningResult:
    def test_basic_creation(self):
        r = ReasoningResult(response="Hello")
        assert r.response == "Hello"
        assert r.thinking is None
        assert r.has_thinking is False

    def test_with_thinking(self):
        r = ReasoningResult(response="42", thinking="Let me calculate...")
        assert r.has_thinking is True

    def test_empty_thinking_not_counted(self):
        r = ReasoningResult(response="42", thinking="   ")
        assert r.has_thinking is False

    def test_str(self):
        r = ReasoningResult(response="Hello world")
        assert str(r) == "Hello world"


class TestGemma4ReasonerInit:
    def test_default_config(self):
        reasoner = Gemma4Reasoner()
        assert reasoner.config.model_size == ModelSize.E4B
        assert reasoner.config.thinking is True

    def test_custom_config(self):
        config = ReasoningConfig(model_size=ModelSize.DENSE_31B, thinking=False)
        reasoner = Gemma4Reasoner(config)
        assert reasoner.config.model_size == ModelSize.DENSE_31B
        assert reasoner.config.thinking is False

    def test_history_empty(self):
        reasoner = Gemma4Reasoner()
        assert len(reasoner.history) == 0

    def test_reset_chat(self, reasoner):
        reasoner._history.append(type("Msg", (), {"role": "user", "content": "hi"})())
        reasoner.reset_chat()
        assert len(reasoner.history) == 0


class TestConfig:
    def test_from_env_defaults(self, monkeypatch):
        # Clear any env vars that might interfere
        for key in ["MODEL_SIZE", "BACKEND", "THINKING", "TEMPERATURE"]:
            monkeypatch.delenv(key, raising=False)
        config = ReasoningConfig.from_env()
        assert config.model_size == ModelSize.E4B
        assert config.thinking is True

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("MODEL_SIZE", "31b")
        monkeypatch.setenv("THINKING", "false")
        config = ReasoningConfig.from_env()
        assert config.model_size == ModelSize.DENSE_31B
        assert config.thinking is False

    def test_invalid_model_size(self, monkeypatch):
        monkeypatch.setenv("MODEL_SIZE", "invalid")
        with pytest.raises(ValueError):
            ReasoningConfig.from_env()

    def test_context_length(self):
        config = ReasoningConfig(model_size=ModelSize.E4B)
        assert config.effective_context_length() == 131_072

        config = ReasoningConfig(model_size=ModelSize.DENSE_31B)
        assert config.effective_context_length() == 262_144

    def test_context_length_capped(self):
        config = ReasoningConfig(
            model_size=ModelSize.DENSE_31B,
            max_context_length=65536,
        )
        assert config.effective_context_length() == 65536
