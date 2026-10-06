"""Aggregate scored records into benchmark metrics, per mode.

Definitions (all rates are correct / total over the stated subset):

* overall_accuracy: every case; answerable cases must match the gold result,
  ambiguous ones must get a clarifying question, unanswerable ones a refusal.
* execution_accuracy: answerable cases whose result contains the gold values
  (matching.compare_values); execution_accuracy_strict also requires the gold
  columns and rows (matching.compare_results).
* first_try_success: answerable cases that reached SQL and whose first query ran
  without errors or failed result checks.
* recovery_rate: of those whose first query failed, the share where a later
  attempt ran cleanly; recovered_and_correct also requires a gold match.
* clarification_accuracy / refusal_accuracy: ambiguous / unanswerable cases with
  the expected status.
* over_caution_rate: answerable cases the agent wrongly declined (clarify/refuse).
"""

import math
from collections import defaultdict
from typing import Any

DECLINED = {"clarification_needed", "unanswerable"}


def rate(correct: int, total: int) -> dict[str, Any]:
    """A rate with its counts; `rate` is None when there is nothing to measure."""
    return {"correct": correct, "total": total, "rate": correct / total if total else None}


def percentile(values: list[float], q: float) -> float | None:
    """Linearly interpolated percentile (q in 0..100), None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for the records of one mode."""
    answerable = [r for r in records if r["expected_behavior"] == "answer"]
    with_sql = [r for r in answerable if r["sql_attempts"] > 0]
    failed_first = [r for r in with_sql if not r["first_try_ok"]]
    ambiguous = [r for r in records if r["expected_behavior"] == "clarify"]
    unanswerable = [r for r in records if r["expected_behavior"] == "refuse"]
    latencies = [r["latency_ms"] for r in records]
    costs = [r["cost_usd"] or 0.0 for r in records]
    count = len(records)

    return {
        "questions": count,
        "overall_accuracy": _share(records, "correct"),
        "execution_accuracy": _share(answerable, "correct"),
        "execution_accuracy_strict": _share(answerable, "strict_match"),
        "first_try_success": _share(with_sql, "first_try_ok"),
        "recovery_rate": rate(
            sum(r["final_ok"] and r["sql_attempts"] > 1 for r in failed_first), len(failed_first)
        ),
        "recovered_and_correct": _share(failed_first, "correct"),
        "clarification_accuracy": _share(ambiguous, "correct"),
        "refusal_accuracy": _share(unanswerable, "correct"),
        "over_caution_rate": rate(
            sum(r["status"] in DECLINED for r in answerable), len(answerable)
        ),
        "by_category": _grouped(records, "category"),
        "by_language": _grouped(records, "language"),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": sum(latencies) / count if count else None,
        },
        "cost_usd": {"total": sum(costs), "per_question": sum(costs) / count if count else None},
        "tokens_per_question": {
            "input": sum(r["input_tokens"] for r in records) / count if count else None,
            "output": sum(r["output_tokens"] for r in records) / count if count else None,
        },
        "llm_calls_per_question": sum(r["llm_calls"] for r in records) / count if count else None,
        "sql_attempts_per_answerable": (
            sum(r["sql_attempts"] for r in answerable) / len(answerable) if answerable else None
        ),
    }


def summarize(records: list[dict[str, Any]], modes: list[str]) -> dict[str, dict[str, Any]]:
    """Metrics per mode, in the given mode order."""
    return {mode: summarize_mode([r for r in records if r["mode"] == mode]) for mode in modes}


def _share(records: list[dict[str, Any]], flag: str) -> dict[str, Any]:
    """Rate of records where `flag` is true."""
    return rate(sum(bool(r[flag]) for r in records), len(records))


def _grouped(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Overall accuracy per value of `key`."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return {name: _share(group, "correct") for name, group in sorted(groups.items())}
