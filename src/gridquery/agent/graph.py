"""The GridQuery LangGraph workflow and the agent's public entry point.

load_context -> understand -+-> clarify ------------> respond_clarification
                            +-> unanswerable -------> respond_unanswerable
                            +-> answer -> write_sql -> execute_sql -> check_result
                                             ^                          |
                                             +---------- retry ---------+
                                               ok -> choose_chart -> explain
                                               give_up -> respond_failure
"""

import logging
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Self
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from gridquery.agent.llm import StructuredLLM, create_llm
from gridquery.agent.nodes import AgentNodes
from gridquery.agent.schemas import AgentResult, AgentStatus, Usage
from gridquery.agent.state import AgentState
from gridquery.agent.tracing import configure_tracing
from gridquery.config import Settings
from gridquery.tools.charts import ChartSpec
from gridquery.tools.schema_context import SchemaContextProvider
from gridquery.tools.sql_executor import SafeSqlExecutor

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")


def build_graph(nodes: AgentNodes) -> CompiledStateGraph:
    """Wire the nodes into the workflow graph."""
    graph = StateGraph(AgentState)
    graph.add_node("load_context", nodes.load_context)
    graph.add_node("understand", nodes.understand)
    graph.add_node("respond_clarification", nodes.respond_clarification)
    graph.add_node("respond_unanswerable", nodes.respond_unanswerable)
    graph.add_node("write_sql", nodes.write_sql)
    graph.add_node("execute_sql", nodes.execute_sql)
    graph.add_node("check_result", nodes.check_result)
    graph.add_node("respond_failure", nodes.respond_failure)
    graph.add_node("choose_chart", nodes.choose_chart)
    graph.add_node("explain", nodes.explain)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "understand")
    graph.add_conditional_edges(
        "understand",
        nodes.route_after_understand,
        {
            "answer": "write_sql",
            "clarify": "respond_clarification",
            "unanswerable": "respond_unanswerable",
        },
    )
    graph.add_edge("write_sql", "execute_sql")
    graph.add_edge("execute_sql", "check_result")
    graph.add_conditional_edges(
        "check_result",
        nodes.route_after_check,
        {"retry": "write_sql", "ok": "choose_chart", "give_up": "respond_failure"},
    )
    graph.add_edge("choose_chart", "explain")
    for terminal in ("explain", "respond_clarification", "respond_unanswerable", "respond_failure"):
        graph.add_edge(terminal, END)
    return graph.compile()


class GridQueryAgent:
    """Answers questions about German electricity data."""

    def __init__(
        self,
        llm: StructuredLLM,
        executor: SafeSqlExecutor,
        *,
        schema: SchemaContextProvider | None = None,
        max_attempts: int = 3,
        explain_max_rows: int = 50,
        max_question_length: int = 1_000,
        today: Callable[[], date] | None = None,
        run_metadata: dict[str, str] | None = None,
    ) -> None:
        """Create an agent from its parts (see from_settings for the usual setup)."""
        self.executor = executor
        self.max_attempts = max_attempts
        self.max_question_length = max_question_length
        self.run_metadata = run_metadata or {}
        self.nodes = AgentNodes(
            llm,
            executor,
            schema or SchemaContextProvider(executor),
            max_attempts=max_attempts,
            explain_max_rows=explain_max_rows,
            today=today or _today_in_germany,
        )
        self.graph = build_graph(self.nodes)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create the agent with the configured LLM, database and limits."""
        configure_tracing(settings)
        return cls(
            create_llm(settings),
            SafeSqlExecutor.from_settings(settings),
            max_attempts=settings.max_sql_attempts,
            explain_max_rows=settings.explain_max_rows,
            max_question_length=settings.max_question_length,
            run_metadata={"provider": settings.llm_provider, "model": settings.llm_model},
        )

    def data_range(self) -> tuple[date, date]:
        """First and last German calendar day with data."""
        return self.nodes.data_range()

    def close(self) -> None:
        """Release the database connection."""
        self.executor.close()

    def ask(self, question: str) -> AgentResult:
        """Answer one question. Never raises; failures come back as status 'failed'."""
        question = question.strip()
        started = time.perf_counter()
        if not question:
            return _failed(question, "Please ask a question.", started)
        if len(question) > self.max_question_length:
            return _failed(
                question,
                f"The question is too long (max {self.max_question_length} characters).",
                started,
            )

        state: AgentState = {"question": question}
        try:
            # Stream so the latest state is still available if a step fails.
            for snapshot in self.graph.stream(
                {"question": question},
                config={
                    "run_name": "gridquery",
                    "tags": ["gridquery"],
                    "metadata": self.run_metadata,
                    "recursion_limit": 10 + 3 * self.max_attempts,
                },
                stream_mode="values",
            ):
                state = snapshot
        except Exception as exc:  # LLM/network errors must not crash the caller.
            logger.exception("Agent run failed")
            return _to_result(
                state,
                started,
                status=AgentStatus.FAILED,
                answer=f"Sorry, something went wrong while answering: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
        return _to_result(state, started)


def _today_in_germany() -> date:
    """Today's date in Germany."""
    return datetime.now(BERLIN).date()


def _to_result(
    state: AgentState,
    started: float,
    *,
    status: AgentStatus | None = None,
    answer: str | None = None,
    error: str | None = None,
) -> AgentResult:
    """Turn the final graph state into an AgentResult."""
    understanding = state.get("understanding")
    result = state.get("result")
    final_status = status or state.get("status", AgentStatus.FAILED)
    answered = final_status is AgentStatus.ANSWERED
    data = result.to_dict() if result is not None and result.ok else None
    chart: ChartSpec | None = state.get("chart") if answered else None
    timings = dict(state.get("timings_ms", {}))
    timings["total"] = (time.perf_counter() - started) * 1000
    return AgentResult(
        question=state.get("question", ""),
        status=final_status,
        language=understanding.language if understanding else "en",
        answer=answer or state.get("answer", ""),
        sql=result.sql if result is not None else None,
        columns=data["columns"] if data else [],
        rows=data["rows"] if data else [],
        row_count=data["row_count"] if data else 0,
        truncated=data["truncated"] if data else False,
        chart=chart,
        assumptions=understanding.assumptions if understanding else [],
        attempts=state.get("attempts", []),
        timings_ms={name: round(ms, 1) for name, ms in timings.items()},
        usage=state.get("usage", Usage()),
        error=error,
    )


def _failed(question: str, message: str, started: float) -> AgentResult:
    """Result for input that never reached the graph."""
    return _to_result({"question": question}, started, status=AgentStatus.FAILED, answer=message)
