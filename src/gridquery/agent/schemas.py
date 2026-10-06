"""Data models of the agent: LLM outputs, attempts, usage and the final result."""

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from gridquery.tools.charts import ChartSpec

Language = Literal["en", "de"]


# --- Structured LLM outputs ------------------------------------------------------------------


class Period(BaseModel):
    """A German calendar period, both days inclusive."""

    start: date = Field(description="First day, YYYY-MM-DD.")
    end: date = Field(description="Last day (inclusive), YYYY-MM-DD.")


class Understanding(BaseModel):
    """Output of the understand step."""

    language: Language = Field(description='"de" if the question is German, otherwise "en".')
    periods: list[Period] = Field(
        default_factory=list,
        description="Resolve first: every period the question asks about, one per compared "
        "period; empty only if the question gives no time hint or asks for patterns/records "
        "over the whole available range.",
    )
    decision: Literal["answer", "clarify", "unanswerable"]
    reason: str = Field(description="One short English sentence explaining the decision.")
    restated_question: str = Field(
        description="Self-contained English question with explicit dates and defaults applied."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Short English list of defaults/interpretations used."
    )
    clarifying_question: str | None = Field(
        default=None,
        description="For 'clarify': one short question, in English if language is 'en', "
        "in German if 'de'.",
    )
    unanswerable_message: str | None = Field(
        default=None,
        description="For 'unanswerable': what is missing and what can be answered instead; "
        "in English if language is 'en', in German if 'de'.",
    )


class ChartSuggestion(BaseModel):
    """The SQL writer's chart for its query: a specification, never code."""

    type: Literal["line", "area", "bar", "stacked_bar", "scatter", "pie", "heatmap", "none"]
    x: str | None = Field(default=None, description="Result column for the x axis / pie slices.")
    y: list[str] = Field(default_factory=list, description="Numeric result column(s).")
    series: str | None = Field(
        default=None,
        description="Optional category column: one line/stack per value, or heatmap rows.",
    )
    title: str | None = Field(default=None, description="Short title in the answer language.")
    x_label: str | None = Field(default=None, description="X axis label in the answer language.")
    y_label: str | None = Field(default=None, description="Y axis label in the answer language.")


class SqlDraft(BaseModel):
    """Output of the write_sql step."""

    sql: str = Field(description="One DuckDB SELECT query.")
    rationale: str = Field(description="One short sentence on how the query answers the question.")
    chart: ChartSuggestion


class Explanation(BaseModel):
    """Output of the explain step."""

    answer: str = Field(description="2-5 sentences in the user's language, without SQL.")


# --- Run records -----------------------------------------------------------------------------


class Attempt(BaseModel):
    """One SQL attempt: what was run and what went wrong, if anything."""

    number: int
    sql: str
    ok: bool
    error: str | None = None
    error_kind: str | None = None
    issues: list[str] = Field(default_factory=list)
    row_count: int = 0
    duration_ms: float = 0.0


class Usage(BaseModel):
    """Token usage and cost of LLM calls."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None = unknown (prices not configured)

    def __add__(self, other: "Usage") -> "Usage":
        """Sum two usage records (cost is unknown if either is unknown)."""
        cost = (
            None
            if self.cost_usd is None or other.cost_usd is None
            else self.cost_usd + other.cost_usd
        )
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


class AgentStatus(StrEnum):
    """How a question was handled."""

    ANSWERED = "answered"
    CLARIFICATION_NEEDED = "clarification_needed"
    UNANSWERABLE = "unanswerable"
    FAILED = "failed"


class AgentResult(BaseModel):
    """Everything the agent returns for one question."""

    question: str
    status: AgentStatus
    language: Language = "en"
    answer: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    chart: ChartSpec | None = None
    assumptions: list[str] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = Field(
        default=None,
        description="Internal error (e.g. LLM/network failure) as 'ExceptionType: message'.",
    )

    @property
    def answer_text(self) -> str:
        """The answer without the SQL block the agent appends (for UIs showing SQL separately)."""
        if self.sql:
            suffix = f"\n\n```sql\n{self.sql.strip()}\n```"
            if self.answer.endswith(suffix):
                return self.answer.removesuffix(suffix)
        return self.answer
