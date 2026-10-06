# GridQuery

An AI data analyst for German electricity data. Ask a question in English or German about
generation by source, consumption or day-ahead prices; GridQuery writes SQL, runs it safely
against a local DuckDB copy of [SMARD](https://www.smard.de) data (Bundesnetzagentur), repairs
its own errors, draws a chart and explains the result. When a question is ambiguous it asks
back; when the data cannot answer it, it says so.

- **Web UI** (Gradio): answer, chart, and a "How the agent worked" section with the SQL, every
  attempt, assumptions, time and cost.
- **HTTP API** (FastAPI): `POST /ask`, `GET /health`, interactive docs at `/docs`.
- **Safety by design**: LLM-written SQL is validated (single read-only `SELECT`, known tables,
  no file/network functions) and runs on a read-only, hardened connection with timeouts and
  row limits. Date coverage is checked in code, not by the model.
- **Measured**: a benchmark of 93 questions with gold answers
  (`uv run python -m gridquery.evaluation`).

## Quick start with Docker

Requirements: Docker with Compose, and an OpenAI API key (or a local Ollama model).

```bash
cp .env.example .env          # then set GRIDQUERY_OPENAI_API_KEY (see Configuration)
docker compose up --build
```

Open **http://localhost:8000** (UI) or **http://localhost:8000/docs** (API).

On the first start the `init` service downloads about three years of SMARD data and builds
`data/gridquery.duckdb` (a few minutes; the download resumes if interrupted). Later starts reuse
the database in `./data`. To refresh the data, delete `data/gridquery.duckdb` and start again.

The port is bound to `127.0.0.1` because every question costs LLM tokens; change the `ports`
entry in `docker-compose.yml` to expose it on your network.

> **Linux hosts:** the container runs as UID 1000. If `./data` belongs to another user, run
> `sudo chown -R 1000 data` once (Docker Desktop on Windows/macOS needs nothing).

## Local setup (without Docker)

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                        # install locked dependencies
cp .env.example .env                           # then fill in your settings
uv run python -m gridquery.ingestion           # download SMARD data into data/raw/ (resumable)
uv run python -m gridquery.database load       # build data/gridquery.duckdb + quality checks
uv run python -m gridquery.api                 # UI + API on http://127.0.0.1:8000
```

Ask from the command line instead:

```bash
uv run python -m gridquery.agent "Wie hoch war der Strompreis im August 2026?"   # add --json
```

## Using the API

```bash
curl -s http://localhost:8000/health
```

```json
{"status": "ok", "provider": "openai", "model": "gpt-4o-mini",
 "data_from": "2023-09-25", "data_to": "2026-09-30", "detail": null}
```

`/health` returns 503 with a `detail` (for example a missing database or API key) when the
agents could not start.

```bash
curl -s -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Which source generated the most electricity in 2024?"}'
```

The response contains:

| Field | Content |
|---|---|
| `status` | `answered`, `clarification_needed` (the answer is a question back), `unanswerable` (the answer explains why) or `failed` |
| `answer` | Explanation in the question's language (without the SQL) |
| `sql` | The final SQL query |
| `results` | `columns`, `rows`, `row_count`, `truncated` |
| `chart` | Chart specification (type, x, y, series, title, labels); rendered by the UI with Plotly |
| `attempts` | Every SQL attempt with its error or failed checks |
| `assumptions` | Interpretations the agent made (e.g. the resolved period) |
| `timings_ms` | Milliseconds per step and `total` |
| `usage` | LLM calls, tokens and cost in USD (if prices are configured) |

Several questions are answered in parallel (`GRIDQUERY_API_MAX_CONCURRENCY`, default 4); more
wait for a free slot.

## Configuration

All settings are environment variables with the `GRIDQUERY_` prefix, read from `.env`
(see `.env.example` for the full list).

| Variable | Default | Purpose |
|---|---|---|
| `GRIDQUERY_LLM_PROVIDER` | `openai` | `openai` (any OpenAI-compatible API) or `ollama` |
| `GRIDQUERY_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `GRIDQUERY_OPENAI_API_KEY` | – | Required for `openai` |
| `GRIDQUERY_OPENAI_BASE_URL` | – | Other OpenAI-compatible providers (Groq, vLLM, …) |
| `GRIDQUERY_OLLAMA_BASE_URL` | `http://localhost:11434` | In Docker use `http://host.docker.internal:11434` |
| `GRIDQUERY_LLM_INPUT_COST_PER_MTOK`, `…_OUTPUT_COST_PER_MTOK` | – | USD per million tokens, for cost reporting and the evaluation spending cap |
| `GRIDQUERY_MAX_SQL_ATTEMPTS` | `3` | SQL attempts per question (self-correction) |
| `GRIDQUERY_QUERY_TIMEOUT_SECONDS` | `10` | Per-query timeout |
| `GRIDQUERY_API_HOST`, `GRIDQUERY_API_PORT` | `127.0.0.1`, `8000` | Server address (Docker uses `0.0.0.0:8000`) |
| `GRIDQUERY_API_MAX_CONCURRENCY` | `4` | Questions answered at the same time |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | `false`, – | Optional LangSmith tracing |

## Evaluation

```bash
uv run python -m gridquery.evaluation               # both modes: with and without self-correction
uv run python -m gridquery.evaluation --resume latest
```

The benchmark (`data/eval/eval_set.jsonl`) has 93 questions: lookups, rankings, comparisons,
calculations, German questions, ambiguous questions (should ask back) and unanswerable ones
(should refuse). Results are saved after every question, so interrupted runs resume without
paying twice; a spending cap (`GRIDQUERY_EVAL_MAX_COST_USD`, default $2) needs the price
settings above. Reports (JSON + Markdown) go to `data/eval/results/`.

Latest run (gpt-4o-mini, 2026-10-06): 64.4% execution accuracy on answerable questions,
92.3% correct refusals, about $0.001 and 5.5 s (median) per question. Known weaknesses are
listed in [docs/errors-and-fixes.md](docs/errors-and-fixes.md).

## Development

```bash
uv run pytest                 # tests (no network; the LLM is mocked)
uv run ruff check .           # lint
uv run ruff format .          # format
```

```
src/gridquery/
  config.py      Typed settings (GRIDQUERY_* environment variables)
  ingestion/     Download and normalize SMARD data
  database/      DuckDB schema, loading, quality checks
  tools/         SQL validation, safe execution, charts
  agent/         LangGraph workflow, prompts, period coverage, agent pool
  evaluation/    Benchmark runner, result matching, metrics, reports
  api/           FastAPI app (also mounts the UI)
  ui/            Gradio UI
tests/           pytest tests, mirroring the package layout
data/            Local data, DuckDB file and evaluation set
```

## Data source

Data: Bundesnetzagentur | SMARD.de, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Times are converted to German
time; energy is in MWh per hour and prices in EUR/MWh (day-ahead auction, bidding zone DE-LU).
