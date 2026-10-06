"""Markdown summary of an evaluation run."""

from collections.abc import Callable
from typing import Any

# (label, metric key) for the headline table; all are rates.
RATE_ROWS = [
    ("Overall accuracy (all cases)", "overall_accuracy"),
    ("Execution accuracy (answerable)", "execution_accuracy"),
    ("Execution accuracy, strict columns/rows", "execution_accuracy_strict"),
    ("First-try success (query ran cleanly)", "first_try_success"),
    ("Recovery rate (failed first query fixed)", "recovery_rate"),
    ("Recovered and correct", "recovered_and_correct"),
    ("Clarification accuracy (ambiguous)", "clarification_accuracy"),
    ("Refusal accuracy (unanswerable)", "refusal_accuracy"),
    ("Over-caution (answerable declined)", "over_caution_rate"),
]

# (label, formatter) for non-rate rows of the headline table.
VALUE_ROWS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("Latency p50", lambda s: _seconds(s["latency_ms"]["p50"])),
    ("Latency p95", lambda s: _seconds(s["latency_ms"]["p95"])),
    ("Cost per question", lambda s: _usd(s["cost_usd"]["per_question"])),
    ("LLM calls per question", lambda s: _num(s["llm_calls_per_question"])),
    ("SQL attempts per answerable", lambda s: _num(s["sql_attempts_per_answerable"])),
    ("Cost total", lambda s: _usd(s["cost_usd"]["total"])),
]


def render_markdown(report: dict[str, Any]) -> str:
    """Render the run report (see __main__.build_report) as Markdown."""
    run, summary = report["run"], report["summary"]
    modes = list(summary)
    lines = [
        f"# GridQuery evaluation {run['run_id']}",
        "",
        f"- Model: `{run['provider']}/{run['model']}`, modes: "
        + ", ".join(
            f"`{name}` ({attempts} SQL attempts)" for name, attempts in run["modes"].items()
        ),
        f"- Dataset: `{run['dataset']}` ({run['cases']} cases)",
        f"- Status: {'complete' if run['complete'] else 'INCOMPLETE: ' + str(run['stop_reason'])}",
        f"- Spent: ${run['spent_usd']:.4f} of ${run['max_cost_usd']:.2f} cap",
        "",
        "## Headline",
        "",
        _header(["Metric", *modes] + (["Δ (pp)"] if len(modes) == 2 else [])),
    ]
    for label, key in RATE_ROWS:
        cells = [_rate(summary[m][key]) for m in modes]
        lines.append(_row([label, *cells] + _delta(summary, modes, key)))
    pad = [""] if len(modes) == 2 else []
    for label, value in VALUE_ROWS:
        lines.append(_row([label, *(value(summary[m]) for m in modes), *pad]))
    for title, key in (("By category", "by_category"), ("By language", "by_language")):
        groups = sorted({g for m in modes for g in summary[m][key]})
        lines += ["", f"## {title}", "", _header([title.split()[-1].title(), "n", *modes])]
        for group in groups:
            stats = [summary[m][key].get(group) for m in modes]
            total = next((s["total"] for s in stats if s), 0)
            lines.append(_row([group, str(total), *(_rate(s) for s in stats)]))

    misses = [r for r in report["results"] if not r["correct"]]
    lines += ["", f"## Misses ({len(misses)})", ""]
    if misses:
        lines.append(_header(["Case", "Mode", "Status", "Reason"]))
        for r in sorted(misses, key=lambda r: (r["case_id"], r["mode"])):
            lines.append(_row([r["case_id"], r["mode"], r["status"], r["match_reason"]]))
    return "\n".join(lines) + "\n"


def _delta(summary: dict[str, Any], modes: list[str], key: str) -> list[str]:
    """Difference of the first mode minus the second, in percentage points."""
    if len(modes) != 2:
        return []
    first, second = (summary[m][key]["rate"] for m in modes)
    if first is None or second is None:
        return ["–"]
    return [f"{100 * (first - second):+.1f}"]


def _rate(stats: dict[str, Any] | None) -> str:
    if not stats or stats["rate"] is None:
        return "–"
    return f"{100 * stats['rate']:.1f}% ({stats['correct']}/{stats['total']})"


def _seconds(ms: float | None) -> str:
    return "–" if ms is None else f"{ms / 1000:.1f} s"


def _usd(value: float | None) -> str:
    return "–" if value is None else f"${value:.4f}"


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def _header(cells: list[str]) -> str:
    return _row(cells) + "\n" + _row(["---"] * len(cells))


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
