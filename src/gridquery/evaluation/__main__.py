"""Command line: `uv run python -m gridquery.evaluation [--resume latest]`.

Runs the benchmark with self-correction enabled and disabled, then writes
`eval_<run_id>.json` (all results and metrics) and `eval_<run_id>.md` (summary
tables) next to the `eval_<run_id>.progress.jsonl` checkpoint.

Answers are judged when the report is built, so after correcting gold answers
`--resume latest` re-scores the run without asking the agent again.
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gridquery.agent.graph import GridQueryAgent
from gridquery.agent.llm import create_llm
from gridquery.agent.tracing import configure_tracing
from gridquery.config import Settings, get_settings
from gridquery.evaluation.dataset import EvalCase, load_cases
from gridquery.evaluation.matching import normalize_text
from gridquery.evaluation.metrics import summarize
from gridquery.evaluation.report import render_markdown
from gridquery.evaluation.runner import (
    AgentFactory,
    Checkpoint,
    Mode,
    ReferenceClock,
    RunOutcome,
    judge_all,
    run_evaluation,
)
from gridquery.logging_config import configure_logging
from gridquery.tools.sql_executor import SafeSqlExecutor

logger = logging.getLogger(__name__)

PROGRESS_SUFFIX = ".progress.jsonl"


def available_modes(settings: Settings) -> dict[str, Mode]:
    """The two configurations compared by default."""
    return {
        "self_correction": Mode("self_correction", settings.max_sql_attempts),
        "single_attempt": Mode("single_attempt", 1),
    }


def agent_factory(settings: Settings) -> AgentFactory:
    """Build real agents (configured LLM and database) for a mode."""

    def build(mode: Mode, clock: ReferenceClock) -> GridQueryAgent:
        return GridQueryAgent(
            create_llm(settings),
            SafeSqlExecutor.from_settings(settings),
            max_attempts=mode.max_attempts,
            explain_max_rows=settings.explain_max_rows,
            max_question_length=settings.max_question_length,
            today=clock,
            run_metadata={
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "eval_mode": mode.name,
            },
        )

    return build


def source_aliases(settings: Settings) -> dict[str, str]:
    """English and German source names -> source key, so "Photovoltaik" matches "solar"."""
    with SafeSqlExecutor.from_settings(settings) as executor:
        result = executor.run("SELECT source, name_en, name_de FROM energy_sources")
    if not result.ok:
        raise SystemExit(f"Cannot read energy_sources: {result.error}")
    return {
        normalize_text(name): normalize_text(source)
        for source, *names in result.rows
        for name in names
    }


def cost_is_known(settings: Settings) -> bool:
    """True if every LLM call gets a cost, which the spending cap needs."""
    return settings.llm_provider == "ollama" or (
        settings.llm_input_cost_per_mtok is not None
        and settings.llm_output_cost_per_mtok is not None
    )


def select_cases(
    cases: list[EvalCase], ids: list[str] | None, categories: list[str] | None, limit: int | None
) -> list[EvalCase]:
    """Filter cases for partial runs."""
    selected = [
        case
        for case in cases
        if (not ids or case.id in ids) and (not categories or case.category in categories)
    ]
    return selected[:limit] if limit else selected


def open_checkpoint(resume: str | None, output_dir: Path, header: dict[str, Any]) -> Checkpoint:
    """Resume an existing checkpoint ("latest" or a path) or start a new one."""
    if resume is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"eval_{run_id}{PROGRESS_SUFFIX}"
        return Checkpoint.create(path, {"run_id": run_id, **header})

    if resume == "latest":
        candidates = sorted(output_dir.glob(f"eval_*{PROGRESS_SUFFIX}"))
        if not candidates:
            raise SystemExit(f"No checkpoint to resume in {output_dir}")
        path = candidates[-1]
    else:
        path = Path(resume)
    checkpoint = Checkpoint.load(path)
    for key in ("provider", "model", "modes"):
        if checkpoint.header.get(key) != header[key]:
            raise SystemExit(
                f"Cannot resume {path}: {key} was {checkpoint.header.get(key)!r}, "
                f"now {header[key]!r}. Start a new run instead."
            )
    logger.info("Resuming %s", path)
    return checkpoint


def build_report(
    checkpoint: Checkpoint,
    cases: list[EvalCase],
    modes: list[Mode],
    outcome: RunOutcome,
    max_cost_usd: float,
    header: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    """Everything about the run in one JSON-serializable dict (`header`: this session's).

    Saved answers are judged here, against the current eval set.
    """
    results = judge_all(cases, checkpoint.results(cases), aliases)
    return {
        "run": {
            "run_id": checkpoint.header["run_id"],
            "started_at": checkpoint.header["started_at"],
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": header["provider"],
            "model": header["model"],
            "modes": header["modes"],
            "dataset": header["dataset"],
            "dataset_sha256": header["dataset_sha256"],
            "cases": len(cases),
            "complete": outcome.complete,
            "stop_reason": outcome.stop_reason,
            "spent_usd": outcome.spent_usd,
            "max_cost_usd": max_cost_usd,
        },
        "summary": summarize(results, [mode.name for mode in modes]),
        "results": results,
    }


def write_report(report: dict[str, Any], checkpoint_path: Path) -> tuple[Path, Path]:
    """Write the JSON and Markdown files next to the checkpoint."""
    stem = checkpoint_path.name.removesuffix(PROGRESS_SUFFIX)
    json_path = checkpoint_path.with_name(f"{stem}.json")
    markdown_path = checkpoint_path.with_name(f"{stem}.md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    """Run (or resume) the evaluation and write the reports."""
    settings = get_settings()
    modes_by_name = available_modes(settings)
    parser = argparse.ArgumentParser(description="Evaluate GridQuery on the benchmark set.")
    parser.add_argument("--dataset", type=Path, default=settings.eval_dataset_path)
    parser.add_argument("--output-dir", type=Path, default=settings.eval_output_dir)
    parser.add_argument(
        "--modes", nargs="+", choices=list(modes_by_name), default=list(modes_by_name)
    )
    parser.add_argument(
        "--max-cost", type=float, default=settings.eval_max_cost_usd, help="Spending cap in USD"
    )
    parser.add_argument("--resume", metavar="PATH|latest", help="Continue an interrupted run")
    parser.add_argument("--ids", nargs="+", help="Only these case ids")
    parser.add_argument("--categories", nargs="+", help="Only these categories")
    parser.add_argument("--limit", type=int, help="Only the first N selected cases")
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)
    if not cost_is_known(settings):
        parser.error(
            "LLM prices are not configured, so the spending cap cannot be enforced. Set "
            "GRIDQUERY_LLM_INPUT_COST_PER_MTOK and GRIDQUERY_LLM_OUTPUT_COST_PER_MTOK."
        )

    cases = select_cases(load_cases(args.dataset), args.ids, args.categories, args.limit)
    if not cases:
        parser.error("No cases selected.")
    modes = [modes_by_name[name] for name in args.modes]
    header = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "modes": {mode.name: mode.max_attempts for mode in modes},
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    checkpoint = open_checkpoint(args.resume, args.output_dir, header)

    configure_tracing(settings)
    aliases = source_aliases(settings)
    outcome = run_evaluation(
        cases,
        modes,
        agent_factory(settings),
        checkpoint,
        max_cost_usd=args.max_cost,
        aliases=aliases,
    )
    report = build_report(checkpoint, cases, modes, outcome, args.max_cost, header, aliases)
    json_path, markdown_path = write_report(report, checkpoint.path)
    sys.stdout.write(render_markdown(report))
    sys.stderr.write(f"\nWrote {json_path} and {markdown_path}\n")
    if not outcome.complete:
        sys.stderr.write(f"Run incomplete ({outcome.stop_reason}); continue with --resume latest\n")
    return 0 if outcome.complete else 1


if __name__ == "__main__":
    sys.exit(main())
