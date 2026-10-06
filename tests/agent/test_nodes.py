"""Each workflow step in isolation, with a mocked LLM and the real safe executor."""

from datetime import date

from agent_helpers import TODAY, draft, explanation, understanding

from gridquery.agent.nodes import apply_period_rules
from gridquery.agent.schemas import (
    AgentStatus,
    Attempt,
    ChartSuggestion,
    Explanation,
    Period,
    SqlDraft,
    Understanding,
)
from gridquery.tools.sql_executor import ErrorKind


def run_sql(nodes, sql: str) -> dict:
    """State after executing and checking one query."""
    state = {"draft": draft(sql), "attempts": []}
    state |= nodes.execute_sql(state)
    return state | nodes.check_result(state)


# --- Step 1: context and understanding --------------------------------------------------------


def test_load_context_provides_schema_and_today(make_nodes) -> None:
    nodes, _ = make_nodes()
    update = nodes.load_context({})
    assert "day_ahead_prices" in update["schema_context"]
    assert update["today"] == TODAY
    assert update["data_period"] == "2026-01-01 to 2026-01-03"
    assert update["data_range"] == (date(2026, 1, 1), date(2026, 1, 3))
    assert update["timings_ms"]["load_context"] >= 0


def test_understand_sends_question_separately_from_rules(make_nodes) -> None:
    nodes, llm = make_nodes(understanding(language="de"))
    state = {
        "question": "Wie hoch war der Verbrauch?",
        "schema_context": "SCHEMA",
        "data_period": "2026-01-01 to 2026-01-02",
        "data_range": (date(2026, 1, 1), date(2026, 1, 2)),
        "today": TODAY,
    }

    update = nodes.understand(state)

    [(system_role, system), (user_role, user)] = llm.calls_for(Understanding)[0]
    assert (system_role, user_role) == ("system", "user")
    assert "SCHEMA" in system and "2026-01-03" in system
    assert "Data is available from 2026-01-01 to 2026-01-02" in system
    assert "The question is written in German" in system
    assert user == "Wie hoch war der Verbrauch?"
    assert update["understanding"].language == "de"
    assert update["usage"].llm_calls == 1


FIRST, LAST = date(2023, 9, 25), date(2026, 9, 30)


def period(start: str, end: str) -> Period:
    return Period(start=date.fromisoformat(start), end=date.fromisoformat(end))


def test_period_inside_the_data_is_answered_unchanged() -> None:
    asked = understanding(periods=[period("2024-01-01", "2024-12-31")])
    assert apply_period_rules(asked, FIRST, LAST) == asked


def test_period_without_data_is_refused_in_code() -> None:
    asked = understanding(periods=[period("2026-10-05", "2026-10-05")])

    result = apply_period_rules(asked, FIRST, LAST)

    assert result.decision == "unanswerable"
    assert result.unanswerable_message == (
        "There is no data for 2026-10-05. Data is available from 2023-09-25 to 2026-09-30; "
        "I can answer questions about that period."
    )


def test_refusal_message_is_in_the_question_language() -> None:
    asked = understanding(language="de", periods=[period("2015-01-01", "2015-12-31")])
    assert apply_period_rules(asked, FIRST, LAST).unanswerable_message == (
        "Für 2015-01-01 bis 2015-12-31 gibt es keine Daten. Daten sind vom 2023-09-25 bis "
        "2026-09-30 verfügbar; Fragen zu diesem Zeitraum kann ich beantworten."
    )


def test_comparison_with_one_period_without_data_is_refused() -> None:
    asked = understanding(
        periods=[period("2023-06-01", "2023-08-31"), period("2024-06-01", "2024-08-31")]
    )
    result = apply_period_rules(asked, FIRST, LAST)
    assert result.decision == "unanswerable"
    assert "2023-06-01 to 2023-08-31" in result.unanswerable_message
    assert "2024" not in result.unanswerable_message.split("Data is")[0]


def test_partly_covered_period_is_answered_with_a_warning() -> None:
    asked = understanding(
        assumptions=["period: 2023"], periods=[period("2023-01-01", "2023-12-31")]
    )

    result = apply_period_rules(asked, FIRST, LAST)

    assert result.decision == "answer"
    assert result.assumptions == [
        "period: 2023",
        "data covers only 2023-09-25 to 2023-12-31 of the requested 2023-01-01 to 2023-12-31; "
        "say so in the answer",
    ]


def test_no_periods_means_whole_range() -> None:
    asked = understanding(periods=[])
    assert apply_period_rules(asked, FIRST, LAST) == asked


def test_clarification_is_kept_when_no_period_is_known() -> None:
    asked = understanding("clarify", periods=[], clarifying_question="Which period?")
    assert apply_period_rules(asked, FIRST, LAST) == asked


def test_clarifying_a_known_period_becomes_an_answer() -> None:
    asked = understanding(
        "clarify",
        periods=[period("2024-03-01", "2024-03-31")],
        clarifying_question="The whole month or a specific week?",
    )
    result = apply_period_rules(asked, FIRST, LAST)
    assert result.decision == "answer"
    assert result.clarifying_question is None


def test_clarifying_a_period_without_data_becomes_a_refusal() -> None:
    asked = understanding("clarify", periods=[period("2015-01-01", "2015-12-31")])
    assert apply_period_rules(asked, FIRST, LAST).decision == "unanswerable"


def test_refusals_for_other_reasons_are_kept() -> None:
    asked = understanding("unanswerable", periods=[period("2024-01-01", "2024-12-31")])
    assert apply_period_rules(asked, FIRST, LAST) == asked


def test_understand_applies_coverage(make_nodes) -> None:
    nodes, _ = make_nodes(understanding(periods=[period("2015-01-01", "2015-12-31")]))
    state = nodes.load_context({}) | {"question": "Solar in 2015?"}
    assert nodes.understand(state)["understanding"].decision == "unanswerable"


def test_route_after_understand_follows_decision(make_nodes) -> None:
    nodes, _ = make_nodes()
    for decision in ("answer", "clarify", "unanswerable"):
        assert nodes.route_after_understand({"understanding": understanding(decision)}) == decision


def test_clarification_uses_model_question_or_localized_fallback(make_nodes) -> None:
    nodes, _ = make_nodes()
    asked = understanding("clarify", clarifying_question="Which month?")
    update = nodes.respond_clarification({"understanding": asked})
    assert update == update | {"answer": "Which month?", "status": AgentStatus.CLARIFICATION_NEEDED}

    fallback = nodes.respond_clarification({"understanding": understanding("clarify", "de")})
    assert fallback["answer"].startswith("Für welchen Zeitraum")


def test_unanswerable_uses_model_message_or_localized_fallback(make_nodes) -> None:
    nodes, _ = make_nodes()
    update = nodes.respond_unanswerable(
        {"understanding": understanding("unanswerable", unanswerable_message="No CO2 data.")}
    )
    assert (update["answer"], update["status"]) == ("No CO2 data.", AgentStatus.UNANSWERABLE)

    fallback = nodes.respond_unanswerable({"understanding": understanding("unanswerable", "de")})
    assert "nicht beantworten" in fallback["answer"]


# --- Step 2: writing, running and checking SQL ------------------------------------------------


def test_write_sql_first_attempt_has_no_feedback(make_nodes) -> None:
    nodes, llm = make_nodes(draft())
    state = {
        "question": "q",
        "understanding": understanding(assumptions=["period: 2026-01-01"]),
        "schema_context": "SCHEMA",
        "today": TODAY,
    }

    update = nodes.write_sql(state)

    user = llm.calls_for(SqlDraft)[0][1][1]
    assert "Total consumption on 2026-01-01." in user
    assert "Assumptions: period: 2026-01-01" in user
    assert "Answer language: English" in user
    assert "Previous attempts" not in user
    assert update["draft"].sql.startswith("SELECT")


def test_write_sql_retry_includes_previous_sql_and_errors(make_nodes) -> None:
    nodes, llm = make_nodes(draft())
    failed = Attempt(number=1, sql="SELECT nope FROM consumption", ok=False, error="Binder Error")
    empty = Attempt(number=2, sql="SELECT 1 WHERE false", ok=True, issues=["no rows"])
    state = {
        "question": "q",
        "understanding": understanding(),
        "schema_context": "S",
        "today": TODAY,
        "attempts": [failed, empty],
    }

    nodes.write_sql(state)

    user = llm.calls_for(SqlDraft)[0][1][1]
    assert "Attempt 1:\nSELECT nope FROM consumption\nProblem: Binder Error" in user
    assert "Attempt 2:\nSELECT 1 WHERE false\nProblem: no rows" in user


def test_execute_and_check_record_a_successful_attempt(make_nodes) -> None:
    nodes, _ = make_nodes()
    state = run_sql(nodes, "SELECT sum(consumption_mwh) AS total_mwh FROM consumption")

    assert state["result"].rows == [(2_400_000.0,)]
    [attempt] = state["attempts"]
    assert (attempt.number, attempt.ok, attempt.row_count, attempt.issues) == (1, True, 1, [])
    assert state["issues"] == []


def test_dangerous_sql_from_the_model_is_blocked_and_recorded(make_nodes) -> None:
    nodes, _ = make_nodes()
    state = run_sql(nodes, "DROP TABLE consumption")

    [attempt] = state["attempts"]
    assert not attempt.ok
    assert attempt.error_kind == ErrorKind.VALIDATION.value
    assert state["issues"][0].startswith("validation error")
    assert nodes.executor.run("SELECT count(*) FROM consumption").rows == [(48,)]


def test_route_after_check(make_nodes) -> None:
    nodes, _ = make_nodes(max_attempts=2)
    ok = run_sql(nodes, "SELECT 1 AS x")
    bad = run_sql(nodes, "SELECT nope FROM consumption")
    implausible = run_sql(nodes, "SELECT -5 AS generation_mwh")

    assert nodes.route_after_check(ok) == "ok"
    assert nodes.route_after_check(bad) == "retry"
    exhausted = bad | {"attempts": bad["attempts"] * 2}
    assert nodes.route_after_check(exhausted) == "give_up"
    # Out of attempts but the query ran: answer, with the issues as warnings.
    assert nodes.route_after_check(implausible | {"attempts": implausible["attempts"] * 2}) == "ok"


def test_respond_failure_is_localized(make_nodes) -> None:
    nodes, _ = make_nodes()
    state = run_sql(nodes, "SELECT nope FROM consumption")
    state["understanding"] = understanding(language="de")

    update = nodes.respond_failure(state)

    assert update["status"] is AgentStatus.FAILED
    assert update["answer"].startswith("Entschuldigung")
    assert "nope" in update["answer"]


# --- Step 3: chart and explanation -------------------------------------------------------------

TIME_SQL = "SELECT date_local, total_generation_mwh, consumption_mwh FROM daily_summary ORDER BY 1"


def chart_for(nodes, sql: str, suggestion: ChartSuggestion, language: str = "en"):
    state = run_sql(nodes, sql) | {
        "draft": draft(sql, suggestion),
        "understanding": understanding(language=language),
    }
    return nodes.choose_chart(state)["chart"]


def test_chart_spec_carries_labels_and_answer_language(make_nodes) -> None:
    nodes, _ = make_nodes()
    suggestion = ChartSuggestion(
        type="line",
        x="date_local",
        y=["consumption_mwh"],
        title="Täglicher Verbrauch",
        x_label="Datum",
        y_label="Verbrauch",
    )
    chart = chart_for(nodes, TIME_SQL, suggestion, language="de")
    assert (chart.title, chart.x_label, chart.y_label, chart.language) == (
        "Täglicher Verbrauch",
        "Datum",
        "Verbrauch",
        "de",
    )


def test_valid_chart_suggestion_is_kept(make_nodes) -> None:
    nodes, _ = make_nodes()
    suggestion = ChartSuggestion(
        type="bar", x="date_local", y=["consumption_mwh"], title="Consumption"
    )
    chart = chart_for(nodes, TIME_SQL, suggestion)
    assert (chart.type, chart.x, chart.y, chart.title) == (
        "bar",
        "date_local",
        ["consumption_mwh"],
        "Consumption",
    )


def test_invalid_chart_suggestion_falls_back_to_rules(make_nodes) -> None:
    nodes, _ = make_nodes()
    suggestion = ChartSuggestion(type="line", x="no_such_column", y=["consumption_mwh"])
    chart = chart_for(nodes, TIME_SQL, suggestion)
    assert (chart.type, chart.x) == ("line", "date_local")
    assert chart.y == ["total_generation_mwh", "consumption_mwh"]


def test_no_chart_when_model_says_none_or_single_value(make_nodes) -> None:
    nodes, _ = make_nodes()
    assert chart_for(nodes, TIME_SQL, ChartSuggestion(type="none")) is None
    single = ChartSuggestion(type="bar", x="x", y=["y"])
    assert chart_for(nodes, "SELECT 'a' AS x, 1 AS y", single) is None


def test_explain_answers_in_question_language_and_appends_sql(make_nodes) -> None:
    nodes, llm = make_nodes(explanation("Der Verbrauch lag bei 2,4 TWh."))
    sql = "SELECT sum(consumption_mwh) AS total_mwh FROM consumption"
    state = run_sql(nodes, sql) | {
        "question": "Wie hoch war der Verbrauch?",
        "understanding": understanding(language="de", assumptions=["period: all data"]),
    }

    update = nodes.explain(state)

    system, user = (content for _, content in llm.calls_for(Explanation)[0])
    assert "in German" in system and "1.234,5" in system
    assert "total_mwh\n2,400,000\n" in user  # full numbers, never scientific notation
    assert "Assumptions: period: all data" in user
    assert update["answer"] == f"Der Verbrauch lag bei 2,4 TWh.\n\n```sql\n{sql}\n```"
    assert update["status"] is AgentStatus.ANSWERED


def test_explain_shows_limited_rows_and_warnings(make_nodes) -> None:
    nodes, llm = make_nodes(explanation())
    state = run_sql(nodes, "SELECT timestamp_utc, price_eur_per_mwh FROM day_ahead_prices")
    state |= {"question": "q", "understanding": understanding(), "issues": ["a warning"]}

    nodes.explain(state)

    user = llm.calls_for(Explanation)[0][1][1]
    assert "... (43 more rows not shown)" in user
    assert "(48 rows)" in user
    assert "Warnings: a warning; Only the first 5 of 48 rows" in user


# --- Language safeguards -----------------------------------------------------------------------


def test_code_detected_language_overrides_the_model(make_nodes) -> None:
    nodes, _ = make_nodes(understanding(language="de"))
    state = {
        "question": "What was the average price last month?",
        "schema_context": "S",
        "data_period": "p",
        "data_range": (date(2026, 1, 1), date(2026, 1, 2)),
        "today": TODAY,
    }
    assert nodes.understand(state)["understanding"].language == "en"


def test_reply_in_wrong_language_is_replaced_by_template(make_nodes) -> None:
    nodes, _ = make_nodes()
    wrong = understanding(
        "clarify", "en", clarifying_question="Für welchen Zeitraum möchten Sie das wissen?"
    )
    assert nodes.respond_clarification({"understanding": wrong})["answer"] == (
        "Could you specify the time period you are interested in?"
    )
    right = understanding("unanswerable", "de", unanswerable_message="Dazu gibt es keine Daten.")
    assert nodes.respond_unanswerable({"understanding": right})["answer"] == (
        "Dazu gibt es keine Daten."
    )


def test_explain_gets_exact_statistics_when_rows_are_cut(make_nodes) -> None:
    nodes, llm = make_nodes(explanation())
    state = run_sql(nodes, "SELECT timestamp_utc, price_eur_per_mwh FROM day_ahead_prices")
    state |= {"question": "q", "understanding": understanding()}

    nodes.explain(state)

    user = llm.calls_for(Explanation)[0][1][1]
    assert "Exact statistics over all 48 rows:" in user
    assert "- price_eur_per_mwh: min -10, max 37, mean 13.5" in user
    assert "- timestamp_utc: from 2026-01-01T00:00:00 to 2026-01-02T23:00:00" in user
