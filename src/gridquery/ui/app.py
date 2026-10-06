"""Gradio UI: ask a question, see the answer and chart, and inspect how it was produced.

`build_ui` takes the function that answers a question, so the UI runs on the
API's agent pool (mounted under /ui by gridquery.api.app) and can be tested
with a fake. The render_* helpers are pure functions from an AgentResult to
display values.
"""

import logging
from collections.abc import Callable
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from gridquery.agent.schemas import AgentResult, AgentStatus
from gridquery.tools.charts import build_figure

logger = logging.getLogger(__name__)

# Checked against the real agent (gpt-4o-mini): correct results, three of them with charts.
EXAMPLES = [
    "Which source generated the most electricity in 2024?",
    "How did the monthly average day-ahead price develop in 2025?",
    "At what hour of the day were prices lowest on average in 2025?",
    "Wie viel Strom wurde im Juni 2026 aus Photovoltaik erzeugt?",
    "Wie hat sich der Stromverbrauch 2025 Monat für Monat entwickelt?",
    "Rangliste der Energiequellen nach Stromerzeugung im Jahr 2025",
]

STATUS_LABELS = {
    AgentStatus.ANSWERED: "Answered",
    AgentStatus.CLARIFICATION_NEEDED: "Clarification needed",
    AgentStatus.UNANSWERABLE: "Cannot be answered from the data",
    AgentStatus.FAILED: "Failed",
}

INTRO = """\
# GridQuery
Ask about German electricity generation, consumption and day-ahead prices, in English or \
German. The agent writes SQL, runs it on [SMARD](https://www.smard.de) data and explains the \
result. Open **How the agent worked** below an answer to see the SQL and every attempt.
"""

Outputs = tuple[str, Any, str, str, Any]


def render_answer(result: AgentResult) -> str:
    """Answer text; non-answers are labelled so they are not mistaken for results."""
    if result.status is AgentStatus.ANSWERED:
        return result.answer_text
    return f"**{STATUS_LABELS[result.status]}.** {result.answer_text}"


def render_details(result: AgentResult) -> str:
    """Markdown summary of the run: status, attempts, assumptions, time, tokens and cost."""
    attempts = len(result.attempts)
    retries = f" ({attempts - 1} retried)" if attempts > 1 else ""
    usage = result.usage
    cost = "unknown" if usage.cost_usd is None else f"${usage.cost_usd:.4f}"
    lines = [
        f"**Status:** {STATUS_LABELS[result.status]} · **Language:** {result.language} · "
        f"**SQL attempts:** {attempts}{retries}",
        f"**Time:** {result.timings_ms.get('total', 0) / 1000:.1f} s · **LLM calls:** "
        f"{usage.llm_calls} · **Tokens:** {usage.input_tokens:,} in / {usage.output_tokens:,} out"
        f" · **Cost:** {cost}",
    ]
    if result.assumptions:
        lines += ["", "**Assumptions**", *(f"- {a}" for a in result.assumptions)]
    if result.attempts:
        lines += ["", "| Attempt | Outcome | Rows | Problem |", "| --- | --- | --- | --- |"]
        for attempt in result.attempts:
            problems = [attempt.error] if attempt.error else []
            problems += attempt.issues
            outcome = "ok" if attempt.ok and not attempt.issues else "retry needed"
            problem = " / ".join(problems).replace("|", "\\|").replace("\n", " ") or "–"
            lines.append(f"| {attempt.number} | {outcome} | {attempt.row_count} | {problem} |")
    return "\n".join(lines)


def make_figure(result: AgentResult) -> go.Figure | None:
    """The chart for the result, or None if there is none or it cannot be drawn."""
    if result.chart is None or not result.rows:
        return None
    try:
        return build_figure(result.chart, result.columns, result.rows)
    except ValueError:
        logger.warning("Chart could not be drawn", exc_info=True)
        return None


def result_table(result: AgentResult) -> pd.DataFrame | None:
    """The result rows as a table, or None if no query result exists."""
    if not result.columns:
        return None
    return pd.DataFrame(result.rows, columns=result.columns)


def respond(ask: Callable[[str], AgentResult], question: str) -> Outputs:
    """Answer one question and turn the result into the UI's output values."""
    if not question.strip():
        raise gr.Error("Please enter a question.")
    try:
        result = ask(question)
    except Exception as exc:  # Show service problems in the UI instead of a stack trace.
        logger.exception("Question failed")
        raise gr.Error(f"The service is not available: {exc}") from exc
    figure = make_figure(result)
    table = result_table(result)
    return (
        render_answer(result),
        gr.Plot(value=figure, visible=figure is not None),
        render_details(result),
        result.sql or "",
        gr.Dataframe(value=table, visible=table is not None),
    )


def build_ui(ask: Callable[[str], AgentResult], *, concurrency: int = 1) -> gr.Blocks:
    """Build the Gradio app.

    Args:
        ask: Answers one question (e.g. the API's agent pool).
        concurrency: Questions answered at the same time (match the pool size).
    """
    with gr.Blocks(title="GridQuery") as blocks:
        gr.Markdown(INTRO)
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. How much solar power was generated in July 2025?",
                scale=5,
                max_lines=3,
                max_length=1_000,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question], label="Examples")
        answer = gr.Markdown(label="Answer")
        chart = gr.Plot(label="Chart", visible=False)
        with gr.Accordion("How the agent worked", open=False):
            details = gr.Markdown()
            sql = gr.Code(label="Final SQL", language="sql", interactive=False)
            table = gr.Dataframe(label="Result data", visible=False, interactive=False)

        outputs = [answer, chart, details, sql, table]
        for trigger in (submit.click, question.submit):
            trigger(
                lambda q: respond(ask, q),
                inputs=question,
                outputs=outputs,
                concurrency_limit=concurrency,
                api_visibility="private",
            )
    return blocks
