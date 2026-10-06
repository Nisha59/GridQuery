"""Graph nodes: one method per workflow step.

Each node reads the state and returns only the fields it changes, plus its own
duration. LLM steps go through the StructuredLLM protocol; SQL goes through the
SafeSqlExecutor, the agent's only way to reach the data.
"""

import logging
import time
from collections.abc import Callable
from datetime import date
from functools import wraps
from typing import Any

from gridquery.agent import prompts
from gridquery.agent.checks import check_result
from gridquery.agent.coverage import check_coverage, describe
from gridquery.agent.language import detect_language
from gridquery.agent.llm import StructuredLLM
from gridquery.agent.schemas import (
    AgentStatus,
    Attempt,
    Explanation,
    Language,
    SqlDraft,
    Understanding,
)
from gridquery.agent.state import AgentState
from gridquery.tools.charts import MAX_Y_COLUMNS, ChartSpec, chart_problems, fallback_chart
from gridquery.tools.schema_context import SchemaContextProvider
from gridquery.tools.sql_executor import SafeSqlExecutor

logger = logging.getLogger(__name__)

Update = dict[str, Any]
Node = Callable[["AgentNodes", AgentState], Update]

FAILURE_MESSAGES: dict[Language, str] = {
    "en": "Sorry, I could not answer this question. The query still failed after "
    "{attempts} attempts. Last error: {error}",
    "de": "Entschuldigung, ich konnte diese Frage nicht beantworten. Die Abfrage schlug auch "
    "nach {attempts} Versuchen fehl. Letzter Fehler: {error}",
}
CLARIFY_FALLBACK: dict[Language, str] = {
    "en": "Could you specify the time period you are interested in?",
    "de": "Für welchen Zeitraum möchten Sie das wissen?",
}
NO_DATA_MESSAGES: dict[Language, str] = {
    "en": "There is no data for {periods}. Data is available from {first} to {last}; "
    "I can answer questions about that period.",
    "de": "Für {periods} gibt es keine Daten. Daten sind vom {first} bis {last} verfügbar; "
    "Fragen zu diesem Zeitraum kann ich beantworten.",
}
UNANSWERABLE_FALLBACK: dict[Language, str] = {
    "en": "I can't answer this from the available data (German electricity generation, "
    "consumption and day-ahead prices).",
    "de": "Das kann ich mit den verfügbaren Daten (Stromerzeugung, Stromverbrauch und "
    "Day-Ahead-Preise in Deutschland) nicht beantworten.",
}


def timed(name: str) -> Callable[[Node], Node]:
    """Record a node's wall-clock duration in timings_ms."""

    def decorator(node: Node) -> Node:
        @wraps(node)
        def wrapper(self: "AgentNodes", state: AgentState) -> Update:
            started = time.perf_counter()
            update = node(self, state)
            update["timings_ms"] = {name: (time.perf_counter() - started) * 1000}
            return update

        return wrapper

    return decorator


class AgentNodes:
    """The workflow steps, sharing the LLM, the SQL executor and the schema context."""

    def __init__(
        self,
        llm: StructuredLLM,
        executor: SafeSqlExecutor,
        schema: SchemaContextProvider,
        *,
        max_attempts: int = 3,
        explain_max_rows: int = 50,
        today: Callable[[], date] = date.today,
    ) -> None:
        """Create the nodes.

        Args:
            llm: Structured LLM used by understand, write_sql and explain.
            executor: Safe SQL executor.
            schema: Provider of the compact schema description.
            max_attempts: Maximum number of SQL attempts per question.
            explain_max_rows: Maximum result rows shown to the LLM when explaining.
            today: Returns the current date in German time.
        """
        self.llm = llm
        self.executor = executor
        self.schema = schema
        self.max_attempts = max_attempts
        self.explain_max_rows = explain_max_rows
        self.today = today
        self._data_range: tuple[date, date] | None = None

    # --- 1. Context and understanding --------------------------------------------------------

    @timed("load_context")
    def load_context(self, state: AgentState) -> Update:
        """Load the schema description, the available period and today's date."""
        first, last = self.data_range()
        return {
            "schema_context": self.schema.text(),
            "data_period": f"{first} to {last}",
            "data_range": (first, last),
            "today": self.today(),
        }

    def data_range(self) -> tuple[date, date]:
        """First and last German calendar day with data."""
        if self._data_range is None:
            with self.executor.internal_cursor() as cursor:
                first, last = cursor.execute(
                    "SELECT min(date_local), max(date_local) FROM daily_summary WHERE hours > 0"
                ).fetchone()
            self._data_range = (first, last)
        return self._data_range

    @timed("understand")
    def understand(self, state: AgentState) -> Update:
        """Detect the language and decide: answer, clarify, or unanswerable.

        The language is detected in code as well; when the text is clearly English or
        German, that wins over the model's guess. The period rules (no clarification once
        a period is known, coverage of the data) are applied in code: see apply_period_rules.
        """
        detected = detect_language(state["question"])
        messages = prompts.understand_messages(
            state["question"],
            state["schema_context"],
            state["today"],
            state["data_period"],
            detected,
        )
        understanding, usage = self.llm.invoke_structured(messages, Understanding)
        if detected and detected != understanding.language:
            logger.info("Language corrected from %s to %s", understanding.language, detected)
            understanding = understanding.model_copy(update={"language": detected})
        understanding = apply_period_rules(understanding, *state["data_range"])
        logger.info(
            "Understood [%s] as %s: %s",
            understanding.language,
            understanding.decision,
            understanding.reason,
        )
        return {"understanding": understanding, "usage": usage}

    @staticmethod
    def route_after_understand(state: AgentState) -> str:
        """Next step after understanding."""
        return state["understanding"].decision

    @timed("respond")
    def respond_clarification(self, state: AgentState) -> Update:
        """Ask the user a clarifying question."""
        understanding = state["understanding"]
        text = _in_language(
            understanding.clarifying_question, understanding.language, CLARIFY_FALLBACK
        )
        return {"answer": text, "status": AgentStatus.CLARIFICATION_NEEDED}

    @timed("respond")
    def respond_unanswerable(self, state: AgentState) -> Update:
        """Explain that the data cannot answer the question."""
        understanding = state["understanding"]
        text = _in_language(
            understanding.unanswerable_message, understanding.language, UNANSWERABLE_FALLBACK
        )
        return {"answer": text, "status": AgentStatus.UNANSWERABLE}

    # --- 2. SQL loop ---------------------------------------------------------------------------

    @timed("write_sql")
    def write_sql(self, state: AgentState) -> Update:
        """Write (or rewrite) the SQL, using feedback from earlier attempts."""
        understanding = state["understanding"]
        messages = prompts.write_sql_messages(
            state["question"],
            understanding.restated_question,
            understanding.assumptions,
            state.get("attempts", []),
            state["schema_context"],
            state["today"],
            understanding.language,
        )
        draft, usage = self.llm.invoke_structured(messages, SqlDraft)
        return {"draft": draft, "usage": usage}

    @timed("execute_sql")
    def execute_sql(self, state: AgentState) -> Update:
        """Run the SQL through the safe executor."""
        return {"result": self.executor.run(state["draft"].sql)}

    @timed("check_result")
    def check_result(self, state: AgentState) -> Update:
        """Look for errors and implausible results; record the attempt."""
        result = state["result"]
        previous = state.get("attempts", [])
        issues = check_result(result, previous)
        attempt = Attempt(
            number=len(previous) + 1,
            sql=result.sql,
            ok=result.ok,
            error=result.error,
            error_kind=result.error_kind.value if result.error_kind else None,
            issues=[] if not result.ok else issues,
            row_count=result.row_count,
            duration_ms=result.execution_time_ms,
        )
        if issues:
            logger.info("Attempt %d has problems: %s", attempt.number, " | ".join(issues))
        return {"attempts": [attempt], "issues": issues}

    def route_after_check(self, state: AgentState) -> str:
        """Retry while attempts remain; afterwards answer if the query ran at all."""
        if not state["issues"]:
            return "ok"
        if len(state["attempts"]) < self.max_attempts:
            return "retry"
        return "ok" if state["result"].ok else "give_up"

    @timed("respond")
    def respond_failure(self, state: AgentState) -> Update:
        """Report that no working query was found."""
        language = state["understanding"].language
        text = FAILURE_MESSAGES[language].format(
            attempts=len(state["attempts"]), error=state["result"].error
        )
        return {"answer": text, "status": AgentStatus.FAILED}

    # --- 3. Presentation -----------------------------------------------------------------------

    @timed("choose_chart")
    def choose_chart(self, state: AgentState) -> Update:
        """Use the SQL writer's chart suggestion if it fits the result, else a rule-based one."""
        result = state["result"]
        rows = result.to_dict()["rows"]
        suggestion = state["draft"].chart
        language = state["understanding"].language
        title = suggestion.title or ""
        if suggestion.type == "none" or result.row_count < 2:
            return {"chart": None}
        if suggestion.x and suggestion.y and len(suggestion.y) <= MAX_Y_COLUMNS:
            spec = ChartSpec(
                type=suggestion.type,
                x=suggestion.x,
                y=suggestion.y,
                series=suggestion.series,
                title=title,
                x_label=suggestion.x_label,
                y_label=suggestion.y_label,
                language=language,
            )
            problems = chart_problems(spec, result.columns, rows)
            if not problems:
                return {"chart": spec}
            logger.info("Chart suggestion rejected: %s", "; ".join(problems))
        chart = fallback_chart(result.columns, rows, title, language)
        logger.info("Using rule-based chart: %s", chart.type if chart else "none")
        return {"chart": chart}

    @timed("explain")
    def explain(self, state: AgentState) -> Update:
        """Explain the result in the question's language and append the SQL."""
        understanding = state["understanding"]
        result = state["result"]
        messages = prompts.explain_messages(
            state["question"],
            understanding.language,
            understanding.assumptions,
            result.sql,
            self._result_table(state),
            self._warnings(state),
        )
        explanation, usage = self.llm.invoke_structured(messages, Explanation)
        answer = f"{explanation.answer.strip()}\n\n```sql\n{result.sql.strip()}\n```"
        return {"answer": answer, "status": AgentStatus.ANSWERED, "usage": usage}

    def _result_table(self, state: AgentState) -> str:
        """Compact text version of the result for the explain prompt."""
        result = state["result"]
        all_rows = result.to_dict()["rows"]
        rows = all_rows[: self.explain_max_rows]
        lines = [" | ".join(result.columns)]
        lines += [" | ".join(_fmt(value) for value in row) for row in rows]
        if result.row_count > len(rows):
            lines.append(f"... ({result.row_count - len(rows)} more rows not shown)")
            lines.append(f"Exact statistics over all {result.row_count} rows:")
            lines += _column_stats(result.columns, all_rows)
        lines.append(f"({result.row_count} rows{', truncated' if result.truncated else ''})")
        return "\n".join(lines)

    def _warnings(self, state: AgentState) -> list[str]:
        """Warnings the explanation should mention."""
        warnings = list(state.get("issues", []))
        result = state["result"]
        if result.row_count > self.explain_max_rows:
            warnings.append(
                f"Only the first {self.explain_max_rows} of {result.row_count} rows are shown "
                "to you; describe the overall result carefully."
            )
        return warnings


def apply_period_rules(understanding: Understanding, first: date, last: date) -> Understanding:
    """Enforce the period rules deterministically; models are unreliable with dates.

    1. "clarify" is only for questions without any time hint. If the model resolved a
       period, it asked about something the question already says: answer instead.
    2. A period without any data turns into an "unanswerable" decision with a message
       from a template; a partly covered period adds an assumption, which the SQL and
       explain steps both see.
    """
    if understanding.decision == "clarify" and understanding.periods:
        logger.info("Period is known; answering instead of asking: %s", understanding.reason)
        understanding = understanding.model_copy(
            update={"decision": "answer", "clarifying_question": None}
        )
    if understanding.decision != "answer":
        return understanding
    coverage = check_coverage(understanding.periods, first, last)
    if not coverage.answerable:
        language = understanding.language
        periods = ", ".join(describe(p, language) for p in coverage.missing)
        logger.info("No data for %s (data: %s to %s)", periods, first, last)
        message = NO_DATA_MESSAGES[language].format(periods=periods, first=first, last=last)
        return understanding.model_copy(
            update={
                "decision": "unanswerable",
                "reason": f"No data for {periods}.",
                "unanswerable_message": message,
            }
        )
    notes = [
        f"data covers only {start} to {end} of the requested {describe(period)}; "
        "say so in the answer"
        for period, start, end in coverage.partial
    ]
    if not notes:
        return understanding
    return understanding.model_copy(update={"assumptions": [*understanding.assumptions, *notes]})


def _in_language(text: str | None, language: Language, fallback: dict[Language, str]) -> str:
    """Use the model's text unless it is missing or clearly in the other language."""
    if not text:
        return fallback[language]
    detected = detect_language(text)
    if detected is not None and detected != language:
        logger.info("Model replied in %s instead of %s; using template", detected, language)
        return fallback[language]
    return text


def _fmt(value: Any) -> str:
    """Format a value for the explain prompt.

    Numbers are written in full with thousands separators (never 1.3e+07), because
    models misjudge magnitudes in scientific notation.
    """
    if value is None:
        return "NULL"
    if isinstance(value, float):
        text = f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _column_stats(columns: list[str], rows: list[list[Any]]) -> list[str]:
    """Min/max/mean per numeric column and min/max per time column, over all rows.

    Lets the explanation describe results it cannot see in full without guessing.
    """
    lines = []
    for index, column in enumerate(columns):
        values = [row[index] for row in rows if row[index] is not None]
        numbers = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
        if numbers and len(numbers) == len(values):
            mean = sum(numbers) / len(numbers)
            lines.append(
                f"- {column}: min {_fmt(float(min(numbers)))}, max {_fmt(float(max(numbers)))}, "
                f"mean {_fmt(mean)}"
            )
        elif values and all(
            isinstance(v, str) and v[:4].isdigit() and v[4:5] == "-" for v in values
        ):
            lines.append(f"- {column}: from {min(values)} to {max(values)}")
    return lines
