"""Command-line entry point: `uv run python -m gridquery.agent "your question"`."""

import argparse
import json
import sys
from pathlib import Path

from gridquery.agent.graph import GridQueryAgent
from gridquery.agent.schemas import AgentResult
from gridquery.config import get_settings
from gridquery.logging_config import configure_logging
from gridquery.tools.charts import build_figure


def render(result: AgentResult) -> str:
    """Human-readable summary of a result."""
    usage = result.usage
    cost = "unknown" if usage.cost_usd is None else f"${usage.cost_usd:.5f}"
    lines = [
        f"[{result.status.value} | {result.language} | attempts: {len(result.attempts)} | "
        f"{result.timings_ms.get('total', 0) / 1000:.1f} s | {usage.llm_calls} LLM calls, "
        f"{usage.input_tokens}+{usage.output_tokens} tokens, cost {cost}]",
        "",
        result.answer,
    ]
    if result.rows:
        lines += ["", " | ".join(result.columns)]
        lines += [" | ".join(str(v) for v in row) for row in result.rows[:10]]
        if result.row_count > 10:
            lines.append(f"... {result.row_count - 10} more rows")
    if result.chart:
        lines += ["", f"Chart: {result.chart.model_dump_json()}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Ask one question and print the answer (or the full JSON result)."""
    parser = argparse.ArgumentParser(description="Ask GridQuery a question.")
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true", help="Print the full structured result")
    parser.add_argument("--chart", type=Path, metavar="FILE.html", help="Save the chart as HTML")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    agent = GridQueryAgent.from_settings(settings)
    try:
        result = agent.ask(args.question)
    finally:
        agent.close()
    if args.chart and result.chart:
        figure = build_figure(result.chart, result.columns, result.rows)
        figure.write_html(args.chart, include_plotlyjs="cdn")
        sys.stderr.write(f"Chart saved to {args.chart}\n")
    output = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    sys.stdout.write((output if args.json else render(result)) + "\n")
    return 0 if result.status.value != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
