"""Helpers for evaluation tests: case builders, agent results and a fake agent."""

from collections.abc import Callable
from datetime import date

from gridquery.agent.schemas import AgentResult, AgentStatus, Attempt, Usage
from gridquery.evaluation.dataset import EvalCase
from gridquery.evaluation.runner import ReferenceClock

REFERENCE_DATE = date(2026, 10, 6)


def case(case_id: str, behavior: str = "answer", **kwargs) -> EvalCase:
    fields = {
        "id": case_id,
        "category": {"answer": "simple_lookup", "clarify": "ambiguous", "refuse": "unanswerable"}[
            behavior
        ],
        "language": "en",
        "question": f"Question {case_id}?",
        "expected_behavior": behavior,
        "reference_date": REFERENCE_DATE,
    }
    if behavior == "answer":
        fields |= {
            "gold_sql": "SELECT 42 AS answer",
            "expected_result": {"columns": ["answer"], "rows": [[42]]},
        }
    return EvalCase(**(fields | kwargs))


def answered(value: float = 42, *, cost: float | None = 0.01, attempts: int = 1) -> AgentResult:
    tries = [Attempt(number=n, sql="SELECT 1", ok=n == attempts) for n in range(1, attempts + 1)]
    return AgentResult(
        question="q",
        status=AgentStatus.ANSWERED,
        answer="It is 42.",
        sql="SELECT 42",
        columns=["x"],
        rows=[[value]],
        row_count=1,
        attempts=tries,
        timings_ms={"total": 1500.0},
        usage=Usage(llm_calls=3, input_tokens=900, output_tokens=90, cost_usd=cost),
    )


def declined(status: AgentStatus, *, cost: float = 0.01) -> AgentResult:
    return AgentResult(
        question="q",
        status=status,
        answer="...",
        timings_ms={"total": 500.0},
        usage=Usage(llm_calls=1, input_tokens=500, output_tokens=50, cost_usd=cost),
    )


def errored(error: str, *, cost: float = 0.0) -> AgentResult:
    return AgentResult(
        question="q",
        status=AgentStatus.FAILED,
        answer="Sorry",
        error=error,
        usage=Usage(cost_usd=cost),
    )


class FakeAgent:
    """Returns scripted results; records questions and the reference date it saw."""

    def __init__(self, respond: Callable[[str], AgentResult], clock: ReferenceClock) -> None:
        self.respond = respond
        self.clock = clock
        self.asked: list[tuple[str, date]] = []
        self.closed = False

    def ask(self, question: str) -> AgentResult:
        self.asked.append((question, self.clock()))
        return self.respond(question)

    def close(self) -> None:
        self.closed = True
