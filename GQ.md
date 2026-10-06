# GridQuery

AI data analyst agent that answers questions in English or German about German
electricity data (generation by source, consumption, prices) from SMARD, the
Bundesnetzagentur's open data platform.

The agent understands a question, writes SQL, runs it safely against DuckDB, repairs
its own errors, creates charts, and explains results in plain language. It asks for
clarification or refuses when a question is ambiguous or can't be answered from the data.

Project goals: production-style engineering, safety by design, measurable evaluation,
and clean code that is easy to explain.

## Package management: uv only

This project uses **uv**. Never use `pip`, `pip install`, or `python -m pip`.

- Add a dependency: `uv add <package>`
- Add a dev tool: `uv add --dev <package>`
- Remove a dependency: `uv remove <package>`
- Install/sync the environment: `uv sync`
- Run anything: `uv run <command>` (e.g. `uv run pytest`, `uv run python -m gridquery.<module>`)

Dependencies live in `pyproject.toml` and `uv.lock`. `uv.lock` is committed.

## Common commands

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff check . --fix     # lint with autofix
uv run ruff format .          # format
uv run python -m gridquery.ingestion           # download SMARD data into data/raw/ (resumable)
uv run python -m gridquery.database load       # rebuild data/gridquery.duckdb + quality checks
uv run python -m gridquery.database check      # quality checks only (read-only)
uv run python -m gridquery.agent "Wie hoch war der Strompreis im August 2026?"  # ask (add --json)
uv run python -m gridquery.api                 # UI + API on http://127.0.0.1:8000
docker compose up --build                      # same, in Docker (builds the DB on first start)
uv run python -m gridquery.evaluation          # benchmark, with/without self-correction (--resume latest)
```

## Data conventions

- All timestamps are UTC in `TIMESTAMP` columns named `*_utc`; each row is the hour starting then.
- Convert to German time with the DuckDB macro `to_berlin_time(ts)`; `*_daily`/`*_monthly`
  views already use German calendar dates.
- Energy in MWh per hour, prices in EUR/MWh. Negative prices are real, never "fix" them.
- The schema lives in `src/gridquery/database/schema.sql`. Table/column descriptions and units live in
  `src/gridquery/database/descriptions.toml` (single source: stored as DuckDB COMMENTs and shown to
  the agent by `tools/schema_context.py`). Update it whenever the schema changes; a test enforces this.

## Stack

- Python 3.11+
- DuckDB: analytical database (single file; the agent only uses read-only connections)
- LangGraph: agent workflow; LangChain: LLM integration; LangSmith: optional tracing
- LLMs: OpenAI-compatible APIs and local Ollama models
- sqlglot: SQL parsing and validation
- Plotly: charts
- FastAPI: HTTP API; Gradio: UI
- pydantic-settings: configuration from `.env`
- pytest, ruff: testing and linting

## Structure

```
src/gridquery/
  config.py      Typed settings (pydantic-settings), GRIDQUERY_* env vars
  ingestion/     Download and normalize SMARD data
  database/      DuckDB schema, loading, read-only connections
  tools/         SQL validation, safe execution, chart creation
  agent/         LangGraph state, nodes, graph, prompts
  evaluation/    Benchmark questions, metrics, reports
  api/           FastAPI app
  ui/            Gradio app
tests/           pytest tests, mirroring the package layout
data/            Local data and DuckDB files (git-ignored)
```

## Coding conventions

- Type hints on all function signatures and public attributes.
- Docstrings (Google style) on all modules, classes, and public functions.
- Small, single-purpose functions; prefer pure functions and pass dependencies in explicitly.
- Use `logging` (`logger = logging.getLogger(__name__)`), never `print` (ruff rule T20 enforces this).
- Read configuration only through `gridquery.config.get_settings()`; never hard-code secrets or paths.
- Keep ruff clean and add tests alongside new code.

## Agent design

- `agent/graph.py` wires the LangGraph workflow: load_context -> understand -> (clarify |
  unanswerable | write_sql -> execute_sql -> check_result [-> retry]) -> choose_chart -> explain.
- Nodes (`agent/nodes.py`) use the `StructuredLLM` protocol (`agent/llm.py`), never LangChain
  directly; tests use a scripted `FakeLLM` (`tests/agent/agent_helpers.py`). No network in tests.
- Deterministic work stays in code (result checks, chart validation, appending the SQL);
  the LLM only understands, writes SQL and explains. The LLM resolves which periods a question
  means (`Understanding.periods`); `nodes.apply_period_rules` decides in code whether the data
  covers them and turns a "clarify" with a known period into an answer (`agent/coverage.py`). Prompts live in `agent/prompts.py`.
- When changing prompts, also try real questions: mocked tests cannot catch prompt quality.
- Charts: the LLM returns a `ChartSpec` (data, never code). Only `tools/charts.build_figure()`
  creates Plotly figures; it validates the spec, adds units from column-name suffixes
  (`_twh`, `_eur_per_mwh`, `_pct`...), strips markup from LLM text and adds the SMARD source.
  New chart types need validation rules, a rendering branch, a CHART_GUIDE entry and tests.

## Evaluation

- `data/eval/eval_set.jsonl` is hand-reviewed: gold SQL + expected result for answerable cases,
  expected behaviour (clarify/refuse) otherwise. Edit it directly; don't regenerate it.
- `evaluation/matching.py` scores execution accuracy (ignores column names, row order, unit
  scale; numeric tolerance = gold precision). Results are checkpointed after every question;
  the spending cap (`GRIDQUERY_EVAL_MAX_COST_USD`) needs the LLM price settings.

## Safety rules

- The agent runs SQL only through `gridquery.tools.sql_executor.SafeSqlExecutor`, never a raw
  DuckDB connection. It validates with `tools/sql_validator.py`, re-checks with DuckDB's parser,
  and runs on a read-only, hardened connection (no external access, locked settings).
- Never loosen the validator's allow/deny lists without adding tests in `tests/tools/`.
- The agent connects to DuckDB with `read_only=True` only.
- All LLM-generated SQL is parsed and validated with sqlglot before execution:
  a single `SELECT` statement, known tables only, no DDL/DML, no file or extension access.
- Enforce query timeouts and row limits from settings.
- Refuse or ask for clarification rather than guess when a question can't be answered from the data.
