"""The LangChain-backed StructuredLLM, tested with a scripted chat model (no network)."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import SecretStr

from gridquery.agent.llm import (
    LangChainStructuredLLM,
    LLMOutputError,
    cost_usd,
    create_chat_model,
    create_llm,
)
from gridquery.agent.schemas import Explanation
from gridquery.config import Settings


class ScriptedChatModel(BaseChatModel):
    """Chat model that returns pre-built messages and records the prompts it got."""

    responses: list[AIMessage]
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_message(args: dict | None, input_tokens: int = 50, output_tokens: int = 10) -> AIMessage:
    tool_calls = [{"name": "Explanation", "args": args, "id": "call_1"}] if args else []
    return AIMessage(
        content="",
        tool_calls=tool_calls,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_parses_output_and_reports_usage_and_cost() -> None:
    model = ScriptedChatModel(responses=[tool_message({"answer": "Hi"})], prompts=[])
    llm = LangChainStructuredLLM(model, input_cost_per_mtok=1.0, output_cost_per_mtok=4.0)

    parsed, usage = llm.invoke_structured([("system", "rules"), ("user", "q")], Explanation)

    assert parsed == Explanation(answer="Hi")
    assert (usage.llm_calls, usage.input_tokens, usage.output_tokens) == (1, 50, 10)
    assert usage.cost_usd == pytest.approx((50 * 1.0 + 10 * 4.0) / 1e6)
    assert [m.type for m in model.prompts[0]] == ["system", "human"]


def test_retries_once_on_invalid_output_and_counts_both_calls() -> None:
    model = ScriptedChatModel(
        responses=[tool_message(None), tool_message({"answer": "ok"})], prompts=[]
    )
    parsed, usage = LangChainStructuredLLM(model).invoke_structured([("user", "q")], Explanation)
    assert parsed.answer == "ok"
    assert usage.llm_calls == 2
    assert usage.cost_usd is None  # prices not configured


def test_raises_after_repeated_invalid_output() -> None:
    model = ScriptedChatModel(responses=[tool_message(None), tool_message(None)], prompts=[])
    with pytest.raises(LLMOutputError, match="Explanation"):
        LangChainStructuredLLM(model).invoke_structured([("user", "q")], Explanation)


def test_cost_calculation() -> None:
    assert cost_usd(1_000_000, 500_000, 0.15, 0.60) == pytest.approx(0.45)
    assert cost_usd(10, 10, None, 0.6) is None


def test_openai_model_requires_api_key() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_chat_model(settings)


def test_openai_model_is_configured_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        llm_model="gpt-test",
        openai_api_key=SecretStr("sk-test"),
        openai_base_url="",  # empty value from .env means "default endpoint"
    )
    model = create_chat_model(settings)
    assert model.model_name == "gpt-test"
    assert model.temperature == 0.0
    assert model.openai_api_base is None


def test_ollama_model_is_free() -> None:
    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="llama3.1")
    llm = create_llm(settings)
    assert llm._input_cost == 0.0 and llm._output_cost == 0.0
