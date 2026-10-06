"""Helpers for agent tests: a scripted fake LLM and output builders.

The test database (tests/conftest.py) has 48 hours from 2026-01-01 00:00 UTC:
solar and lignite at 100 MWh each, consumption 50,000 MWh, prices -10 ... 37 EUR/MWh.
"""

from datetime import date

from pydantic import BaseModel

from gridquery.agent.llm import Messages
from gridquery.agent.schemas import (
    ChartSuggestion,
    Explanation,
    SqlDraft,
    Understanding,
    Usage,
)

TODAY = date(2026, 1, 3)
CALL_USAGE = Usage(llm_calls=1, input_tokens=100, output_tokens=20, cost_usd=0.001)


class FakeLLM:
    """StructuredLLM that returns scripted outputs per schema and records every call."""

    def __init__(self, *outputs: BaseModel | Exception, usage: Usage = CALL_USAGE) -> None:
        self.queues: dict[type, list[BaseModel | Exception]] = {}
        for output in outputs:
            schema = getattr(output, "schema_for", type(output))
            self.queues.setdefault(schema, []).append(output)
        self.usage = usage
        self.calls: list[tuple[type, Messages]] = []

    def invoke_structured(self, messages: Messages, schema: type) -> tuple[BaseModel, Usage]:
        self.calls.append((schema, messages))
        queue = self.queues.get(schema)
        if not queue:
            raise AssertionError(f"Unexpected LLM call for {schema.__name__}")
        output = queue.pop(0)
        if isinstance(output, Exception):
            raise output
        return output, self.usage

    def calls_for(self, schema: type) -> list[Messages]:
        return [messages for called, messages in self.calls if called is schema]


class ScriptedLLMError(Exception):
    """An exception scripted for one schema."""

    def __init__(self, schema: type, message: str = "LLM unavailable") -> None:
        super().__init__(message)
        self.schema_for = schema


def understanding(
    decision: str = "answer",
    language: str = "en",
    restated: str = "Total consumption on 2026-01-01.",
    **kwargs,
) -> Understanding:
    return Understanding(
        language=language,
        decision=decision,
        reason="test",
        restated_question=restated,
        **kwargs,
    )


def draft(
    sql: str = "SELECT sum(consumption_mwh) AS total_mwh FROM consumption",
    chart: ChartSuggestion | None = None,
) -> SqlDraft:
    return SqlDraft(sql=sql, rationale="test", chart=chart or ChartSuggestion(type="none"))


def explanation(text: str = "Consumption was 2.4 TWh.") -> Explanation:
    return Explanation(answer=text)
