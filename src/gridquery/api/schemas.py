"""Request and response models of the HTTP API."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from gridquery.agent.schemas import AgentResult, AgentStatus, Attempt, Language, Usage
from gridquery.tools.charts import ChartSpec


class AskRequest(BaseModel):
    """A question for the agent."""

    question: str = Field(min_length=1, examples=["What was the average price in March 2024?"])


class QueryResults(BaseModel):
    """The rows returned by the final SQL query."""

    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = Field(description="True if the result was cut at the row limit.")


class AskResponse(BaseModel):
    """The agent's answer and everything needed to show how it was produced."""

    question: str
    status: AgentStatus = Field(
        description="answered, clarification_needed (answer is a question back), "
        "unanswerable (answer explains why) or failed."
    )
    language: Language
    answer: str = Field(description="Answer text in the question's language, without SQL.")
    sql: str | None = Field(description="Final SQL query, if one was run.")
    results: QueryResults
    chart: ChartSpec | None = Field(description="Chart specification (data, not an image).")
    assumptions: list[str]
    attempts: list[Attempt] = Field(description="Every SQL attempt, including failed ones.")
    timings_ms: dict[str, float] = Field(description="Milliseconds per step; 'total' overall.")
    usage: Usage

    @classmethod
    def from_result(cls, result: AgentResult) -> "AskResponse":
        """Build the response from the agent's result."""
        return cls(
            question=result.question,
            status=result.status,
            language=result.language,
            answer=result.answer_text,
            sql=result.sql,
            results=QueryResults(
                columns=result.columns,
                rows=result.rows,
                row_count=result.row_count,
                truncated=result.truncated,
            ),
            chart=result.chart,
            assumptions=result.assumptions,
            attempts=result.attempts,
            timings_ms=result.timings_ms,
            usage=result.usage,
        )


class HealthResponse(BaseModel):
    """Service status."""

    status: Literal["ok", "unavailable"]
    provider: str
    model: str
    data_from: date | None = None
    data_to: date | None = None
    detail: str | None = Field(default=None, description="Why the service is unavailable.")
