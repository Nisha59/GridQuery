"""Run the benchmark: ask every case in every mode and score the answers.

Every finished answer is appended to a checkpoint file (JSONL) immediately, so
an interrupted run resumes where it stopped and never pays for an answer twice.
A spending cap stops the run before the next question would exceed it. Cases
alternate between modes, so a run stopped early still compares modes fairly.

Infrastructure errors (network, rate limits, invalid API key) are not scored as
wrong answers: they are not saved as results and are retried on resume. After
a few in a row the run stops, since the remaining questions would fail too.
"""

import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from gridquery.agent.schemas import AgentResult, AgentStatus
from gridquery.evaluation.dataset import EvalCase
from gridquery.evaluation.matching import compare_results, compare_values, has_limit

logger = logging.getLogger(__name__)

MAX_SAVED_ROWS = 100  # more than any gold result has
# Errors from the model itself (bad structured output) count as wrong answers.
MODEL_ERROR_TYPES = ("LLMOutputError",)
EXPECTED_STATUS = {
    "clarify": AgentStatus.CLARIFICATION_NEEDED,
    "refuse": AgentStatus.UNANSWERABLE,
}


@dataclass(frozen=True)
class Mode:
    """One agent configuration to evaluate.

    Attributes:
        name: Label used in reports, e.g. "self_correction".
        max_attempts: SQL attempts per question (1 disables self-correction).
    """

    name: str
    max_attempts: int


class Agent(Protocol):
    """The part of GridQueryAgent the runner uses."""

    def ask(self, question: str) -> AgentResult:
        """Answer one question."""
        ...


class ReferenceClock:
    """A settable "today" so relative dates resolve as on each case's reference date."""

    def __init__(self, today: date) -> None:
        """Start at `today`."""
        self.today = today

    def __call__(self) -> date:
        """Return the current reference date."""
        return self.today


AgentFactory = Callable[[Mode, ReferenceClock], Agent]


@dataclass(frozen=True)
class RunOutcome:
    """How a run ended.

    Attributes:
        complete: True if every case was answered in every mode.
        stop_reason: Why the run stopped early, if it did.
        spent_usd: Total cost recorded in the checkpoint, including earlier sessions.
    """

    complete: bool
    stop_reason: str | None
    spent_usd: float


class SpendingCapError(RuntimeError):
    """Cost cannot be tracked, so the spending cap cannot be enforced."""


# --- Recording and judging ------------------------------------------------------------------


def record_result(case: EvalCase, mode: Mode, result: AgentResult) -> dict[str, Any]:
    """What the agent did, as a JSON-serializable checkpoint line (not yet judged).

    Judging happens later (see `judge`), so corrected gold answers never need a re-run.
    """
    attempts = result.attempts
    return {
        "type": "result",
        "mode": mode.name,
        "case_id": case.id,
        "case_fingerprint": case.fingerprint(),
        "status": result.status.value,
        "sql_attempts": len(attempts),
        "first_try_ok": bool(attempts) and attempts[0].ok and not attempts[0].issues,
        "final_ok": bool(attempts) and attempts[-1].ok and not attempts[-1].issues,
        "latency_ms": result.timings_ms.get("total", 0.0),
        "llm_calls": result.usage.llm_calls,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "cost_usd": result.usage.cost_usd,
        "question": case.question,
        "answer": result.answer,
        "sql": result.sql,
        "columns": result.columns,
        "rows": result.rows[:MAX_SAVED_ROWS],
        "row_count": result.row_count,
        "attempts": [a.model_dump(mode="json") for a in attempts],
        "error": result.error,
    }


def judge(
    case: EvalCase, record: dict[str, Any], aliases: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The record plus the verdict against the current version of the case.

    Adds: category, language, expected_behavior, correct (value match for answerable
    cases, expected status otherwise), strict_match, match_reason, column_map.
    """
    verdict = {
        "category": case.category,
        "language": case.language,
        "expected_behavior": case.expected_behavior,
        "strict_match": None,
        "column_map": {},
    }
    status = record["status"]
    if case.expected_behavior != "answer":
        expected = EXPECTED_STATUS[case.expected_behavior].value
        ok = status == expected
        reason = "expected behaviour" if ok else f"expected {expected}, got {status}"
        return record | verdict | {"correct": ok, "match_reason": reason}
    if status != AgentStatus.ANSWERED.value:
        reason = f"expected an answer, got {status}"
        return record | verdict | {"correct": False, "strict_match": False, "match_reason": reason}

    assert case.expected_result is not None
    gold = case.expected_result
    truncated = record["row_count"] > len(record["rows"])
    if truncated and not (case.gold_sql and has_limit(case.gold_sql)):
        reason = f"row count {record['row_count']} != expected {len(gold.rows)}"
        return record | verdict | {"correct": False, "strict_match": False, "match_reason": reason}
    options = {
        "required_columns": case.required_columns,
        "gold_sql": case.gold_sql,
        "aliases": aliases,
    }
    values = compare_values(gold.columns, gold.rows, record["rows"], **options)
    strict = compare_results(gold.columns, gold.rows, record["columns"], record["rows"], **options)
    return (
        record
        | verdict
        | {
            "correct": values.match,
            "strict_match": strict.match,
            "match_reason": values.reason if not values.match else strict.reason,
            "column_map": strict.column_map,
        }
    )


def judge_all(
    cases: Iterable[EvalCase],
    records: Iterable[dict[str, Any]],
    aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Judge saved records against their (current) cases."""
    by_id = {case.id: case for case in cases}
    return [judge(by_id[r["case_id"]], r, aliases) for r in records]


def is_infrastructure_error(result: AgentResult) -> bool:
    """True if the run failed for reasons outside the model's control."""
    return result.error is not None and not result.error.startswith(MODEL_ERROR_TYPES)


# --- Checkpoint ------------------------------------------------------------------------------


class Checkpoint:
    """Append-only JSONL file: a header line, then one line per answered (mode, case)."""

    def __init__(self, path: Path, header: dict[str, Any], lines: list[dict[str, Any]]) -> None:
        """Use `create` or `load` instead."""
        self.path = path
        self.header = header
        self._lines = lines

    @classmethod
    def create(cls, path: Path, header: dict[str, Any]) -> "Checkpoint":
        """Start a new checkpoint file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"type": "header", **header}
        checkpoint = cls(path, header, [])
        checkpoint._write(header)
        return checkpoint

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Open an existing checkpoint to resume it."""
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines or lines[0].get("type") != "header":
            raise ValueError(f"{path} is not an evaluation checkpoint")
        return cls(path, lines[0], lines[1:])

    @property
    def spent_usd(self) -> float:
        """Everything paid so far, including discarded (infrastructure-error) calls."""
        return sum(line.get("cost_usd") or 0.0 for line in self._lines)

    def results(self, cases: Iterable[EvalCase]) -> list[dict[str, Any]]:
        """Saved results that are still valid for the current version of each case."""
        current = {case.id: case.fingerprint() for case in cases}
        return [
            line
            for line in self._lines
            if line["type"] == "result" and current.get(line["case_id"]) == line["case_fingerprint"]
        ]

    def append(self, line: dict[str, Any]) -> None:
        """Save one line durably."""
        self._lines.append(line)
        self._write(line)

    def _write(self, line: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


# --- Run -------------------------------------------------------------------------------------


def run_evaluation(
    cases: list[EvalCase],
    modes: list[Mode],
    agent_factory: AgentFactory,
    checkpoint: Checkpoint,
    *,
    max_cost_usd: float,
    max_consecutive_errors: int = 3,
    aliases: Mapping[str, str] | None = None,
) -> RunOutcome:
    """Answer all cases not yet in the checkpoint, in every mode.

    Args:
        cases: Benchmark cases.
        modes: Agent configurations to compare.
        agent_factory: Builds an agent for a mode, using the clock as "today".
        checkpoint: Where results are saved (and earlier results are read from).
        max_cost_usd: Stop before the next question would push the total past this.
        max_consecutive_errors: Stop after this many infrastructure errors in a row.
        aliases: String aliases for judging (only used for progress logging here).

    Raises:
        SpendingCapError: If a result has unknown cost (prices not configured).
    """
    done = {(r["mode"], r["case_id"]) for r in checkpoint.results(cases)}
    todo = [(case, mode) for case in cases for mode in modes if (mode.name, case.id) not in done]
    logger.info("%d of %d answers to go", len(todo), len(cases) * len(modes))
    if not todo:
        return RunOutcome(True, None, checkpoint.spent_usd)

    clock = ReferenceClock(cases[0].reference_date)
    agents = {mode.name: agent_factory(mode, clock) for mode in modes}
    consecutive_errors = 0
    try:
        for number, (case, mode) in enumerate(todo, start=1):
            spent = checkpoint.spent_usd
            estimate = _average_cost(checkpoint.results(cases))
            if spent + estimate > max_cost_usd:
                reason = (
                    f"spending cap reached: ${spent:.4f} spent, next question ~${estimate:.4f}, "
                    f"cap ${max_cost_usd:.2f}"
                )
                logger.warning("Stopping: %s", reason)
                return RunOutcome(False, reason, spent)

            clock.today = case.reference_date
            result = agents[mode.name].ask(case.question)
            if result.usage.cost_usd is None:
                raise SpendingCapError(
                    "LLM cost is unknown; set GRIDQUERY_LLM_INPUT_COST_PER_MTOK and "
                    "GRIDQUERY_LLM_OUTPUT_COST_PER_MTOK so the spending cap can be enforced."
                )

            if is_infrastructure_error(result):
                checkpoint.append(_discarded(case, mode, result))
                consecutive_errors += 1
                logger.warning("%s/%s: infrastructure error: %s", mode.name, case.id, result.error)
                if consecutive_errors >= max_consecutive_errors:
                    reason = f"{consecutive_errors} infrastructure errors in a row: {result.error}"
                    return RunOutcome(False, reason, checkpoint.spent_usd)
                continue

            consecutive_errors = 0
            record = record_result(case, mode, result)
            checkpoint.append(record)
            verdict = judge(case, record, aliases)
            logger.info(
                "[%d/%d] %-15s %-10s %-22s %s  %.1fs $%.4f",
                number,
                len(todo),
                mode.name,
                case.id,
                record["status"],
                "OK  " if verdict["correct"] else "MISS",
                record["latency_ms"] / 1000,
                record["cost_usd"],
            )
    finally:
        for agent in agents.values():
            close = getattr(agent, "close", None)
            if close:
                close()

    remaining = {(c.id, m.name) for c in cases for m in modes} - {
        (r["case_id"], r["mode"]) for r in checkpoint.results(cases)
    }
    if remaining:
        return RunOutcome(False, f"{len(remaining)} answers missing (errors)", checkpoint.spent_usd)
    return RunOutcome(True, None, checkpoint.spent_usd)


def _average_cost(results: list[dict[str, Any]]) -> float:
    """Mean cost per answered question so far (0 before the first)."""
    costs = [r["cost_usd"] or 0.0 for r in results]
    return sum(costs) / len(costs) if costs else 0.0


def _discarded(case: EvalCase, mode: Mode, result: AgentResult) -> dict[str, Any]:
    """Checkpoint line for a failed call: keeps the spend, is not a result."""
    return {
        "type": "discarded",
        "mode": mode.name,
        "case_id": case.id,
        "error": result.error,
        "cost_usd": result.usage.cost_usd,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
