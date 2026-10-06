"""Aggregating scored records into metrics."""

import pytest

from gridquery.evaluation.metrics import percentile, summarize, summarize_mode


def record(
    behavior: str = "answer",
    *,
    correct: bool = True,
    status: str = "answered",
    attempts: int = 1,
    first_ok: bool = True,
    final_ok: bool = True,
    category: str = "simple_lookup",
    language: str = "en",
    mode: str = "self_correction",
    latency_ms: float = 1000.0,
    cost: float = 0.001,
) -> dict:
    return {
        "mode": mode,
        "expected_behavior": behavior,
        "correct": correct,
        "strict_match": correct,
        "status": status,
        "sql_attempts": attempts,
        "first_try_ok": first_ok,
        "final_ok": final_ok,
        "category": category,
        "language": language,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "input_tokens": 1000,
        "output_tokens": 100,
        "llm_calls": 3,
    }


def test_percentile_interpolates() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_summary_rates() -> None:
    records = [
        record(),  # correct on first try
        record(attempts=2, first_ok=False, final_ok=True, correct=True),  # recovered
        record(attempts=3, first_ok=False, final_ok=True, correct=False),  # fixed, wrong
        record(attempts=3, first_ok=False, final_ok=False, correct=False, status="failed"),
        record(attempts=0, correct=False, status="unanswerable"),  # over-cautious
        record("clarify", status="clarification_needed", attempts=0, category="ambiguous"),
        record("clarify", status="answered", correct=False, category="ambiguous", language="de"),
        record("refuse", status="unanswerable", attempts=0, category="unanswerable"),
    ]

    summary = summarize_mode(records)

    assert summary["questions"] == 8
    assert summary["overall_accuracy"]["rate"] == pytest.approx(4 / 8)
    assert summary["execution_accuracy"] == {"correct": 2, "total": 5, "rate": 0.4}
    assert summary["first_try_success"] == {"correct": 1, "total": 4, "rate": 0.25}
    assert summary["recovery_rate"]["correct"] == 2 and summary["recovery_rate"]["total"] == 3
    assert summary["recovered_and_correct"]["correct"] == 1
    assert summary["clarification_accuracy"]["rate"] == 0.5
    assert summary["refusal_accuracy"]["rate"] == 1.0
    assert summary["over_caution_rate"]["correct"] == 1
    assert summary["by_category"]["ambiguous"] == {"correct": 1, "total": 2, "rate": 0.5}
    assert summary["by_language"]["de"]["rate"] == 0.0
    assert summary["cost_usd"]["total"] == pytest.approx(0.008)
    assert summary["latency_ms"]["p50"] == 1000.0


def test_missing_subsets_have_no_rate() -> None:
    summary = summarize_mode([record("refuse", status="unanswerable", attempts=0)])
    assert summary["recovery_rate"]["rate"] is None
    assert summary["clarification_accuracy"]["rate"] is None


def test_summarize_splits_by_mode() -> None:
    records = [record(), record(mode="single_attempt", correct=False)]
    summary = summarize(records, ["self_correction", "single_attempt"])
    assert list(summary) == ["self_correction", "single_attempt"]
    assert summary["single_attempt"]["execution_accuracy"]["rate"] == 0.0
