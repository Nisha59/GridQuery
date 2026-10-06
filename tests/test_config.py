"""Smoke tests for the configuration module."""

import pytest

from gridquery.config import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load with safe defaults when no environment is set."""
    monkeypatch.delenv("GRIDQUERY_LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.max_sql_attempts >= 1


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables with the GRIDQUERY_ prefix override defaults."""
    monkeypatch.setenv("GRIDQUERY_LLM_PROVIDER", "ollama")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
