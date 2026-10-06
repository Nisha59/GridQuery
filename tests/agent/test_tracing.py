import os

import pytest
from pydantic import SecretStr

from gridquery.agent.tracing import configure_tracing
from gridquery.config import Settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


def test_tracing_off_by_default() -> None:
    assert configure_tracing(Settings(_env_file=None)) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_on_exports_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "gq-test")
    configured = Settings(_env_file=None)

    assert configure_tracing(configured) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
    assert os.environ["LANGSMITH_PROJECT"] == "gq-test"


def test_tracing_without_key_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    configured = Settings(_env_file=None)
    assert configured.langsmith_api_key is None
    assert configure_tracing(configured) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_api_key_is_secret() -> None:
    configured = Settings(_env_file=None, openai_api_key=SecretStr("sk-secret"))
    assert "sk-secret" not in repr(configured)
