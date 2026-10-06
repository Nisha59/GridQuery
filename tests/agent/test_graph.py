"""End-to-end runs of the LangGraph workflow with a mocked LLM."""

import pytest
from agent_helpers import ScriptedLLMError, draft, explanation, understanding

from gridquery.agent.llm import LLMOutputError
from gridquery.agent.schemas import (
    AgentStatus,
    ChartSuggestion,
    Explanation,
    SqlDraft,
    Understanding,
    Usage,
)

DAILY_SQL = "SELECT date_local, consumption_mwh FROM daily_summary ORDER BY date_local"
LINE = ChartSuggestion(type="line", x="date_local", y=["consumption_mwh"], title="Daily")


def test_answered_question_returns_full_structured_result(make_agent) -> None:
    agent, llm = make_agent(
        understanding(assumptions=["period: all data"]),
        draft(DAILY_SQL, LINE),
        explanation("Daily consumption was about 1.2 TWh."),
    )

    result = agent.ask("How much electricity was used per day?")

    assert result.status is AgentStatus.ANSWERED
    assert result.language == "en"
    assert result.answer.startswith("Daily consumption was about 1.2 TWh.")
    assert result.answer.endswith(f"```sql\n{DAILY_SQL}\n```")
    assert result.sql == DAILY_SQL
    assert result.columns == ["date_local", "consumption_mwh"]
    assert result.rows[0] == ["2026-01-01", 1_150_000.0]  # 23 hours on Jan 1 in Berlin time
    assert result.row_count == 3 and not result.truncated
    assert result.chart is not None and result.chart.type == "line"
    assert result.assumptions == ["period: all data"]
    assert len(result.attempts) == 1 and result.attempts[0].ok
    assert result.usage == Usage(llm_calls=3, input_tokens=300, output_tokens=60, cost_usd=0.003)
    assert {
        "load_context",
        "understand",
        "write_sql",
        "execute_sql",
        "check_result",
        "choose_chart",
        "explain",
        "total",
    } <= result.timings_ms.keys()
    assert result.error is None
    assert result.model_dump(mode="json")  # serializable for the API


def test_sql_error_is_fixed_on_retry(make_agent) -> None:
    agent, llm = make_agent(
        understanding(),
        draft("SELECT consumption_gwh FROM consumption"),  # column does not exist
        draft(),
        explanation(),
    )

    result = agent.ask("Total consumption?")

    assert result.status is AgentStatus.ANSWERED
    first, second = result.attempts
    assert (first.ok, first.error_kind) == (False, "execution")
    assert "consumption_gwh" in first.error
    assert second.ok and second.number == 2
    retry_prompt = llm.calls_for(SqlDraft)[1][1][1]
    assert "Attempt 1:\nSELECT consumption_gwh FROM consumption" in retry_prompt
    assert result.usage.llm_calls == 4


def test_dangerous_sql_is_blocked_then_fixed(make_agent, executor) -> None:
    agent, _ = make_agent(understanding(), draft("DELETE FROM consumption"), draft(), explanation())

    result = agent.ask("Total consumption?")

    assert result.status is AgentStatus.ANSWERED
    assert result.attempts[0].error_kind == "validation"
    assert executor.run("SELECT count(*) FROM consumption").rows == [(48,)]


def test_empty_result_is_retried_and_accepted_when_confirmed(make_agent) -> None:
    empty_sql = "SELECT * FROM day_ahead_prices WHERE price_eur_per_mwh > 1000"
    agent, _ = make_agent(
        understanding(), draft(empty_sql), draft(empty_sql + ";"), explanation("No hours matched.")
    )

    result = agent.ask("Hours above 1000 EUR/MWh?")

    assert result.status is AgentStatus.ANSWERED
    assert [a.row_count for a in result.attempts] == [0, 0]
    assert result.attempts[0].issues and not result.attempts[1].issues
    assert result.rows == []


def test_gives_up_after_max_attempts(make_agent) -> None:
    bad = draft("SELECT nope FROM consumption")
    agent, llm = make_agent(understanding(language="de"), bad, bad, max_attempts=2)

    result = agent.ask("Wie hoch war der Verbrauch?")

    assert result.status is AgentStatus.FAILED
    assert result.answer.startswith("Entschuldigung")
    assert len(result.attempts) == 2
    assert result.chart is None and result.rows == []
    assert not llm.calls_for(Explanation)


def test_implausible_result_is_answered_with_warning_after_max_attempts(make_agent) -> None:
    implausible = draft("SELECT -5.0 AS generation_mwh")
    agent, llm = make_agent(
        understanding(), implausible, implausible, explanation(), max_attempts=2
    )

    result = agent.ask("Generation?")

    assert result.status is AgentStatus.ANSWERED
    explain_prompt = llm.calls_for(Explanation)[0][1][1]
    assert "Warnings: Column generation_mwh has negative energy values" in explain_prompt


def test_clarifying_question_skips_sql(make_agent) -> None:
    agent, llm = make_agent(
        understanding("clarify", clarifying_question="For which period? E.g. January 2026?")
    )

    result = agent.ask("How much solar power was generated?")

    assert result.status is AgentStatus.CLARIFICATION_NEEDED
    assert result.answer == "For which period? E.g. January 2026?"
    assert result.sql is None and result.attempts == []
    assert result.usage.llm_calls == 1


def test_unanswerable_question_is_declined(make_agent) -> None:
    agent, _ = make_agent(
        understanding("unanswerable", "de", unanswerable_message="Zu Importen gibt es keine Daten.")
    )

    result = agent.ask("Wie viel Strom hat Deutschland importiert?")

    assert result.status is AgentStatus.UNANSWERABLE
    assert (result.language, result.answer) == ("de", "Zu Importen gibt es keine Daten.")


def test_german_question_gets_german_explanation(make_agent) -> None:
    agent, llm = make_agent(
        understanding(language="de"), draft(), explanation("Der Verbrauch betrug 2,4 TWh.")
    )

    result = agent.ask("Wie hoch war der Stromverbrauch?")

    assert result.language == "de"
    assert result.answer.startswith("Der Verbrauch betrug 2,4 TWh.")
    assert "in German" in llm.calls_for(Explanation)[0][0][1]


@pytest.mark.parametrize("failing", [Understanding, SqlDraft, Explanation])
def test_llm_failure_returns_failed_result_with_partial_records(make_agent, failing) -> None:
    outputs = [understanding(), draft(), explanation()]
    outputs = [ScriptedLLMError(failing) if type(o) is failing else o for o in outputs]
    agent, _ = make_agent(*outputs)

    result = agent.ask("Total consumption?")

    assert result.status is AgentStatus.FAILED
    assert "LLM unavailable" in result.answer
    assert result.error == "ScriptedLLMError: LLM unavailable"
    assert "total" in result.timings_ms


def test_invalid_structured_output_is_reported(make_agent) -> None:
    error = LLMOutputError("Model returned no valid Understanding")
    error.schema_for = Understanding
    agent, _ = make_agent(error)

    result = agent.ask("Total consumption?")

    assert result.status is AgentStatus.FAILED
    assert "no valid Understanding" in result.answer
    assert result.error is not None and result.error.startswith("LLMOutputError")


@pytest.mark.parametrize("question", ["", "   ", "x" * 1_001])
def test_invalid_questions_never_reach_the_llm(make_agent, question: str) -> None:
    agent, llm = make_agent()
    result = agent.ask(question)
    assert result.status is AgentStatus.FAILED
    assert llm.calls == []


def test_unknown_cost_stays_unknown(make_agent) -> None:
    agent, _ = make_agent(
        understanding(),
        draft(),
        explanation(),
        usage=Usage(llm_calls=1, input_tokens=10, output_tokens=5, cost_usd=None),
    )
    result = agent.ask("Total consumption?")
    assert result.usage.cost_usd is None
    assert result.usage.input_tokens == 30


def test_many_attempts_stay_within_recursion_limit(make_agent) -> None:
    bad = draft("SELECT nope FROM consumption")
    agent, _ = make_agent(understanding(), *[bad] * 6, max_attempts=6)
    result = agent.ask("q")
    assert result.status is AgentStatus.FAILED
    assert len(result.attempts) == 6


def test_answer_text_strips_only_the_appended_sql(make_agent) -> None:
    agent, _ = make_agent(understanding(), draft(DAILY_SQL, LINE), explanation("It was 1.2 TWh."))
    result = agent.ask("How much electricity was used per day?")
    assert result.answer_text == "It was 1.2 TWh."
    assert result.model_copy(update={"answer": "No SQL here."}).answer_text == "No SQL here."
