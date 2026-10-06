"""Loading the eval set and running the command line end to end (with a fake agent)."""

import json
from pathlib import Path

import pytest
from eval_helpers import answered, case, declined

import gridquery.evaluation.__main__ as cli
from gridquery.agent.schemas import AgentStatus
from gridquery.config import Settings
from gridquery.evaluation.dataset import EvalCase, load_cases

EVAL_SET = Path(__file__).parents[2] / "data" / "eval" / "eval_set.jsonl"


def write_cases(path: Path, cases: list[EvalCase]) -> Path:
    path.write_text(
        "".join(c.model_dump_json(exclude_none=True) + "\n" for c in cases), encoding="utf-8"
    )
    return path


@pytest.mark.skipif(not EVAL_SET.exists(), reason="eval set not present")
def test_eval_set_is_valid() -> None:
    cases = load_cases(EVAL_SET)
    assert 80 <= len(cases) <= 100
    assert {c.expected_behavior for c in cases} == {"answer", "clarify", "refuse"}
    assert {c.language for c in cases} == {"en", "de"}


def test_answerable_case_needs_gold() -> None:
    with pytest.raises(ValueError, match="expected_result"):
        case("a", expected_result=None)
    with pytest.raises(ValueError, match="required_columns"):
        case("a", required_columns=["nope"])


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    path = write_cases(tmp_path / "set.jsonl", [case("a"), case("a")])
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(path)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    dataset = write_cases(tmp_path / "set.jsonl", [case("a"), case("b", "refuse")])
    settings = Settings(
        _env_file=None,
        eval_dataset_path=dataset,
        eval_output_dir=tmp_path / "results",
        llm_input_cost_per_mtok=0.15,
        llm_output_cost_per_mtok=0.6,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "source_aliases", lambda s: {})
    return settings


def test_cli_writes_json_and_markdown_and_resumes(settings, fake_factory, monkeypatch) -> None:
    responses = {"Question a?": answered(42), "Question b?": declined(AgentStatus.UNANSWERABLE)}
    factory = fake_factory(lambda q: responses[q])
    monkeypatch.setattr(cli, "agent_factory", lambda s: factory)

    assert cli.main([]) == 0

    [json_path] = settings.eval_output_dir.glob("eval_*Z.json")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["run"]["complete"]
    summary = report["summary"]
    assert list(summary) == ["self_correction", "single_attempt"]
    assert summary["self_correction"]["overall_accuracy"]["rate"] == 1.0
    markdown = json_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "| Execution accuracy (answerable) | 100.0% (1/1) | 100.0% (1/1) | +0.0 |" in markdown
    assert "| Refusal accuracy (unanswerable) | 100.0% (1/1) |" in markdown

    again = fake_factory(lambda q: responses[q])
    monkeypatch.setattr(cli, "agent_factory", lambda s: again)
    assert cli.main(["--resume", "latest"]) == 0
    assert again.agents == {}  # nothing asked twice


def test_cli_refuses_to_run_without_prices(settings, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_input_cost_per_mtok", None)
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_refuses_to_resume_with_another_model(settings, fake_factory, monkeypatch) -> None:
    monkeypatch.setattr(cli, "agent_factory", lambda s: fake_factory(lambda q: answered()))
    cli.main(["--ids", "a"])
    monkeypatch.setattr(settings, "llm_model", "another-model")
    with pytest.raises(SystemExit, match="model"):
        cli.main(["--resume", "latest"])


def test_source_aliases_come_from_the_database(db_path) -> None:
    aliases = cli.source_aliases(Settings(_env_file=None, duckdb_path=db_path))
    assert aliases["solar (photovoltaics)"] == "solar"
    assert aliases["photovoltaik"] == "solar"
    assert aliases["braunkohle"] == "lignite"
