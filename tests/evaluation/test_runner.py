"""Running the benchmark: scoring, checkpoints, resuming, spending cap, errors."""

from datetime import date

import pytest
from eval_helpers import answered, case, declined, errored

from gridquery.agent.schemas import AgentStatus
from gridquery.evaluation.runner import (
    MAX_SAVED_ROWS,
    Checkpoint,
    Mode,
    SpendingCapError,
    judge,
    judge_all,
    record_result,
    run_evaluation,
)

MODES = [Mode("self_correction", 3), Mode("single_attempt", 1)]


@pytest.fixture
def checkpoint(tmp_path) -> Checkpoint:
    return Checkpoint.create(tmp_path / "eval.progress.jsonl", {"run_id": "test"})


def run(cases, factory, checkpoint, *, max_cost_usd: float = 10.0, modes=MODES):
    return run_evaluation(cases, modes, factory, checkpoint, max_cost_usd=max_cost_usd)


# --- Recording and judging ------------------------------------------------------------------


def verdict(c, result, aliases=None) -> dict:
    return judge(c, record_result(c, MODES[0], result), aliases)


def test_judge_answer_against_gold() -> None:
    good = verdict(case("a"), answered(42))
    bad = verdict(case("a"), answered(41))
    assert good["correct"] and good["strict_match"] and good["match_reason"] == "match"
    assert not bad["correct"] and not bad["strict_match"]
    assert good["first_try_ok"] and good["final_ok"] and good["sql_attempts"] == 1


def test_record_keeps_recovery_details() -> None:
    record = record_result(case("a"), MODES[0], answered(42, attempts=2))
    assert not record["first_try_ok"] and record["final_ok"] and record["sql_attempts"] == 2


@pytest.mark.parametrize(
    ("behavior", "status", "correct"),
    [
        ("clarify", AgentStatus.CLARIFICATION_NEEDED, True),
        ("clarify", AgentStatus.UNANSWERABLE, False),
        ("refuse", AgentStatus.UNANSWERABLE, True),
        ("refuse", AgentStatus.CLARIFICATION_NEEDED, False),
    ],
)
def test_judge_expected_behaviour(behavior: str, status: AgentStatus, correct: bool) -> None:
    assert verdict(case("x", behavior), declined(status))["correct"] is correct


def test_declining_an_answerable_question_is_wrong() -> None:
    result = verdict(case("a"), declined(AgentStatus.CLARIFICATION_NEEDED))
    assert not result["correct"]
    assert "expected an answer" in result["match_reason"]


def test_judge_uses_the_current_gold_answer() -> None:
    record = record_result(case("a"), MODES[0], answered(41))
    corrected = case("a", expected_result={"columns": ["answer"], "rows": [[41]]})
    assert not judge(case("a"), record)["correct"]
    assert judge(corrected, record)["correct"]


def test_judge_uses_aliases_for_labels() -> None:
    c = case(
        "a",
        expected_result={"columns": ["source", "twh"], "rows": [["solar", 63.1]]},
    )
    result = answered()
    result.columns, result.rows = ["name", "twh"], [["Solar (photovoltaics)", 63.1]]
    assert not verdict(c, result)["strict_match"]
    assert verdict(c, result, {"solar (photovoltaics)": "solar"})["strict_match"]


def test_truncated_long_result_fails_on_row_count() -> None:
    result = answered()
    result.rows, result.row_count = [[42]] * MAX_SAVED_ROWS, MAX_SAVED_ROWS + 265
    judged = verdict(case("a"), result)
    assert not judged["correct"]
    assert "row count 365" in judged["match_reason"]


# --- Running ---------------------------------------------------------------------------------


def test_runs_every_case_in_every_mode_and_saves_each_result(fake_factory, checkpoint) -> None:
    cases = [case("a"), case("b", "clarify")]
    factory = fake_factory(lambda q: answered(42))

    outcome = run(cases, factory, checkpoint)

    assert outcome.complete and outcome.stop_reason is None
    results = Checkpoint.load(checkpoint.path).results(cases)
    assert {(r["case_id"], r["mode"]) for r in results} == {
        (c, m.name) for c in ("a", "b") for m in MODES
    }
    assert outcome.spent_usd == pytest.approx(0.04)
    assert all(agent.closed for agent in factory.agents.values())


def test_agent_sees_each_cases_reference_date(fake_factory, checkpoint) -> None:
    cases = [case("a", reference_date=date(2025, 3, 1)), case("b")]
    factory = fake_factory(lambda q: answered())
    run(cases, factory, checkpoint, modes=MODES[:1])
    assert [d for _, d in factory.agents["self_correction"].asked] == [
        date(2025, 3, 1),
        date(2026, 10, 6),
    ]


def test_interrupted_run_resumes_without_asking_again(fake_factory, checkpoint) -> None:
    cases = [case("a"), case("b"), case("c")]
    calls = []

    def flaky(question: str):
        calls.append(question)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return answered()

    with pytest.raises(KeyboardInterrupt):
        run(cases, fake_factory(flaky), checkpoint, modes=MODES[:1])

    resumed = Checkpoint.load(checkpoint.path)
    assert len(resumed.results(cases)) == 2
    factory = fake_factory(lambda q: answered())
    outcome = run(cases, factory, resumed, modes=MODES[:1])
    assert outcome.complete
    assert [q for q, _ in factory.agents["self_correction"].asked] == ["Question c?"]


def test_nothing_to_do_builds_no_agents(fake_factory, checkpoint) -> None:
    cases = [case("a")]
    run(cases, fake_factory(lambda q: answered()), checkpoint)
    factory = fake_factory(lambda q: answered())
    assert run(cases, factory, Checkpoint.load(checkpoint.path)).complete
    assert factory.agents == {}


def test_corrected_gold_answer_is_not_asked_again(fake_factory, checkpoint) -> None:
    run([case("a")], fake_factory(lambda q: answered(41)), checkpoint, modes=MODES[:1])
    corrected = [case("a", expected_result={"columns": ["answer"], "rows": [[41]]})]
    factory = fake_factory(lambda q: answered())
    assert run(corrected, factory, checkpoint, modes=MODES[:1]).complete
    assert factory.agents == {}
    [judged] = judge_all(corrected, checkpoint.results(corrected))
    assert judged["correct"]


def test_edited_question_is_asked_again(fake_factory, checkpoint) -> None:
    run([case("a")], fake_factory(lambda q: answered()), checkpoint, modes=MODES[:1])
    edited = [case("a", question="A better question?")]
    factory = fake_factory(lambda q: answered())
    run(edited, factory, checkpoint, modes=MODES[:1])
    assert [q for q, _ in factory.agents["self_correction"].asked] == ["A better question?"]


def test_spending_cap_stops_before_exceeding(fake_factory, checkpoint) -> None:
    cases = [case(str(i)) for i in range(10)]
    outcome = run(
        cases, fake_factory(lambda q: answered(cost=0.01)), checkpoint, max_cost_usd=0.035
    )

    assert not outcome.complete
    assert "spending cap" in outcome.stop_reason
    assert outcome.spent_usd == pytest.approx(0.03)  # a 4th answer would exceed 0.035


def test_spending_cap_counts_earlier_sessions(fake_factory, checkpoint) -> None:
    cases = [case(str(i)) for i in range(4)]
    run(cases[:2], fake_factory(lambda q: answered(cost=0.01)), checkpoint, modes=MODES[:1])
    outcome = run(
        cases,
        fake_factory(lambda q: answered(cost=0.01)),
        Checkpoint.load(checkpoint.path),
        modes=MODES[:1],
        max_cost_usd=0.025,
    )
    assert not outcome.complete
    assert outcome.spent_usd == pytest.approx(0.02)


def test_unknown_cost_stops_the_run(fake_factory, checkpoint) -> None:
    with pytest.raises(SpendingCapError):
        run([case("a")], fake_factory(lambda q: answered(cost=None)), checkpoint)


def test_infrastructure_errors_are_not_scored_and_stop_the_run(fake_factory, checkpoint) -> None:
    cases = [case(str(i)) for i in range(5)]
    factory = fake_factory(lambda q: errored("RateLimitError: quota exceeded", cost=0.001))

    outcome = run(cases, factory, checkpoint, modes=MODES[:1])

    assert not outcome.complete
    assert "3 infrastructure errors in a row" in outcome.stop_reason
    assert checkpoint.results(cases) == []
    assert outcome.spent_usd == pytest.approx(0.003)  # money spent is still counted


def test_single_infrastructure_error_is_retried_on_resume(fake_factory, checkpoint) -> None:
    cases = [case("a"), case("b")]
    responses = iter([errored("APIConnectionError: reset"), answered()])
    outcome = run(cases, fake_factory(lambda q: next(responses)), checkpoint, modes=MODES[:1])
    assert not outcome.complete and "missing" in outcome.stop_reason

    factory = fake_factory(lambda q: answered())
    assert run(cases, factory, Checkpoint.load(checkpoint.path), modes=MODES[:1]).complete
    assert [q for q, _ in factory.agents["self_correction"].asked] == ["Question a?"]


def test_invalid_model_output_counts_as_wrong_answer(fake_factory, checkpoint) -> None:
    cases = [case("a")]
    factory = fake_factory(lambda q: errored("LLMOutputError: no valid SqlDraft"))
    outcome = run(cases, factory, checkpoint, modes=MODES[:1])
    assert outcome.complete
    [result] = judge_all(cases, checkpoint.results(cases))
    assert not result["correct"] and result["status"] == "failed"
