"""Gradio UI helpers, tested without starting a server."""

import gradio as gr
import pytest

from gridquery.agent.schemas import AgentResult, AgentStatus, Attempt, Usage
from gridquery.tools.charts import ChartSpec
from gridquery.ui.app import (
    EXAMPLES,
    build_ui,
    make_figure,
    render_answer,
    render_details,
    respond,
    result_table,
)

SQL = "SELECT source, generation_twh FROM t"


def result(**overrides) -> AgentResult:
    fields = {
        "question": "q",
        "status": AgentStatus.ANSWERED,
        "answer": f"Wind led with 112.6 TWh.\n\n```sql\n{SQL}\n```",
        "sql": SQL,
        "columns": ["source", "generation_twh"],
        "rows": [["Wind onshore", 112.56], ["Solar", 63.15]],
        "row_count": 2,
        "chart": ChartSpec(type="bar", x="source", y=["generation_twh"]),
        "attempts": [
            Attempt(number=1, sql="SELECT x", ok=False, error="Binder Error: column x | y"),
            Attempt(number=2, sql=SQL, ok=True, row_count=2),
        ],
        "timings_ms": {"total": 4200.0},
        "usage": Usage(llm_calls=4, input_tokens=5200, output_tokens=310, cost_usd=0.0011),
    }
    return AgentResult(**(fields | overrides))


def test_examples_cover_both_languages() -> None:
    assert len(EXAMPLES) >= 4
    assert any("Wie" in q or "Welche" in q or "An welchem" in q for q in EXAMPLES)
    assert any(q.startswith(("Which", "How", "What", "At")) for q in EXAMPLES)


def test_answer_hides_the_sql_block() -> None:
    assert render_answer(result()) == "Wind led with 112.6 TWh."


def test_non_answers_are_labelled() -> None:
    clarify = result(status=AgentStatus.CLARIFICATION_NEEDED, answer="Which year?", sql=None)
    assert render_answer(clarify) == "**Clarification needed.** Which year?"


def test_details_show_attempts_problems_and_cost() -> None:
    details = render_details(result(assumptions=["period: 2024"]))
    assert "**SQL attempts:** 2 (1 retried)" in details
    assert "| 1 | retry needed | 0 | Binder Error: column x \\| y |" in details
    assert "| 2 | ok | 2 | – |" in details
    assert "- period: 2024" in details
    assert "$0.0011" in details and "5,200 in" in details


def test_unknown_cost_is_shown_as_unknown() -> None:
    details = render_details(result(usage=Usage(cost_usd=None)))
    assert "**Cost:** unknown" in details


def test_figure_is_built_from_the_chart_spec() -> None:
    figure = make_figure(result())
    assert figure is not None and figure.data[0].type == "bar"


def test_no_figure_without_chart_or_for_invalid_spec() -> None:
    assert make_figure(result(chart=None)) is None
    invalid = ChartSpec(type="line", x="missing", y=["generation_twh"])
    assert make_figure(result(chart=invalid)) is None


def test_result_table() -> None:
    table = result_table(result())
    assert list(table.columns) == ["source", "generation_twh"] and len(table) == 2
    assert result_table(result(columns=[], rows=[])) is None


def test_respond_produces_all_outputs() -> None:
    answer, chart, details, sql, table = respond(lambda q: result(question=q), "Most in 2024?")
    assert answer == "Wind led with 112.6 TWh."
    assert chart.visible is True
    assert "SQL attempts" in details
    assert sql == SQL
    assert table.visible is True


def test_respond_reports_service_problems_and_empty_questions() -> None:
    def broken(question: str) -> AgentResult:
        raise RuntimeError("starting up")

    with pytest.raises(gr.Error):
        respond(broken, "q")
    with pytest.raises(gr.Error):
        respond(broken, "   ")


def test_build_ui(monkeypatch) -> None:
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    assert isinstance(build_ui(lambda q: result(), concurrency=2), gr.Blocks)
