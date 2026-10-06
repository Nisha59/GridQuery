"""LLM access for the agent.

Nodes depend only on the small StructuredLLM protocol: "give me an instance of
this Pydantic model for these messages, and tell me how many tokens it took".
The production implementation wraps a LangChain chat model (OpenAI-compatible
API or Ollama); tests use a scripted fake. This keeps nodes easy to test and
independent of LangChain internals.
"""

import logging
from typing import Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from gridquery.agent.schemas import Usage
from gridquery.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Messages = list[tuple[str, str]]  # (role, content), role in {"system", "user", "assistant"}


class LLMOutputError(Exception):
    """The model did not return valid structured output."""


class StructuredLLM(Protocol):
    """Anything that turns messages into a validated Pydantic object."""

    def invoke_structured(self, messages: Messages, schema: type[T]) -> tuple[T, Usage]:
        """Return the parsed output and the token usage of producing it."""
        ...


class LangChainStructuredLLM:
    """StructuredLLM backed by a LangChain chat model using tool/function calling."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        input_cost_per_mtok: float | None = None,
        output_cost_per_mtok: float | None = None,
        max_parse_retries: int = 1,
    ) -> None:
        """Wrap a chat model.

        Args:
            chat_model: LangChain chat model that supports tool calling.
            input_cost_per_mtok: USD per million input tokens (None = unknown).
            output_cost_per_mtok: USD per million output tokens (None = unknown).
            max_parse_retries: Extra calls when the output does not match the schema.
        """
        self._chat_model = chat_model
        self._input_cost = input_cost_per_mtok
        self._output_cost = output_cost_per_mtok
        self._max_parse_retries = max_parse_retries

    def invoke_structured(self, messages: Messages, schema: type[T]) -> tuple[T, Usage]:
        """Call the model and parse its output into `schema`, retrying once on bad output.

        Raises:
            LLMOutputError: If no valid output was produced.
        """
        runnable = self._chat_model.with_structured_output(
            schema, method="function_calling", include_raw=True
        )
        usage = Usage(cost_usd=0.0)
        error: object = None
        for attempt in range(self._max_parse_retries + 1):
            output = runnable.invoke(messages)
            usage = usage + self._usage(output.get("raw"))
            parsed = output.get("parsed")
            if isinstance(parsed, schema):
                return parsed, usage
            error = output.get("parsing_error") or "no structured output"
            logger.warning(
                "%s: invalid structured output (attempt %d): %s",
                schema.__name__,
                attempt + 1,
                error,
            )
        raise LLMOutputError(f"Model returned no valid {schema.__name__}: {error}")

    def _usage(self, message: object) -> Usage:
        """Token usage and cost of one raw model message."""
        metadata = getattr(message, "usage_metadata", None) or {}
        input_tokens = int(metadata.get("input_tokens", 0))
        output_tokens = int(metadata.get("output_tokens", 0))
        return Usage(
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(input_tokens, output_tokens, self._input_cost, self._output_cost),
        )


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_mtok: float | None,
    output_cost_per_mtok: float | None,
) -> float | None:
    """Cost of one call in USD, or None if prices are not configured."""
    if input_cost_per_mtok is None or output_cost_per_mtok is None:
        return None
    return (input_tokens * input_cost_per_mtok + output_tokens * output_cost_per_mtok) / 1e6


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the configured LangChain chat model (OpenAI-compatible or Ollama)."""
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )

    from langchain_openai import ChatOpenAI

    if settings.openai_api_key is None:
        raise ValueError("GRIDQUERY_OPENAI_API_KEY is not set (see .env.example).")
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )


def create_llm(settings: Settings) -> LangChainStructuredLLM:
    """Build the structured LLM from settings; local Ollama models cost nothing."""
    is_local = settings.llm_provider == "ollama"
    return LangChainStructuredLLM(
        create_chat_model(settings),
        input_cost_per_mtok=0.0 if is_local else settings.llm_input_cost_per_mtok,
        output_cost_per_mtok=0.0 if is_local else settings.llm_output_cost_per_mtok,
    )
