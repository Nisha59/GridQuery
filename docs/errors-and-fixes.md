# Errors and fixes: evaluation set, evaluation pipeline, agent fixes

Log of every problem found while building the evaluation set and pipeline and while
fixing the agent (2026-10-06), with its cause and fix. Grouped by where the problem
was; the last section lists what is still open.

Runs referenced below (in `data/eval/results/`):

| Run | What | Execution accuracy (self-correction) |
|---|---|---|
| `eval_20261006T112401Z` | Baseline, before agent fixes (re-scored on the current eval set) | 31.5% (23/73) |
| `eval_20261006T121553Z` | After fixes 1 and 2 | 64.4% (47/73) |

---

## 1. Agent errors found by the evaluation

### 1.1 False refusals: "Data for 2024 is not available"
- **Symptom:** 31 of 69 answerable questions declined in the baseline, e.g.
  "How much electricity did Germany consume in 2024?" → *"The data does not include
  information for 2024."* Reproduced with `python -m gridquery.agent`, so it is not an
  eval-harness artefact.
- **Cause:** the understand prompt gave the data range (2023-09-25 to 2026-09-30) and asked
  the model to decide whether a period is covered. gpt-4o-mini gets this date comparison wrong.
  The prompt also listed "periods outside the available data" as a refusal reason, which the
  model over-applied.
- **Fix:** the model only *resolves* periods; code decides coverage.
  - `Understanding.periods: list[Period]` (`agent/schemas.py`), one per compared period.
  - `agent/coverage.py`: `check_coverage()` → missing / partly covered periods.
  - `nodes.apply_period_rules()`: no data → "unanswerable" with a templated message (EN/DE)
    naming the available range; partly covered (e.g. 2023) → answered, with a warning added
    to the assumptions.
  - Prompt: "Do NOT decide whether a period has data… Never use 'unanswerable' because of the
    time period."
- **Result:** answerable-but-declined dropped from 46.6% to 9.6%.

### 1.2 "Yesterday" answered with an empty result
- **Symptom:** "How much electricity was generated yesterday?" (2026-10-05, after the data
  ends) ran a query, got 0 rows, and replied "No data matched" with status *answered*.
- **Fix:** covered by 1.1: the period has no data, so code refuses before any SQL runs.

### 1.3 Wrong result shape: daily rows for yearly totals
- **Symptom:** "Total offshore wind generation in 2025?" returned 365 daily rows plus
  `LIMIT 500`; the explanation computed the total from column statistics. Same for shares,
  averages and comparisons (365/500/731 rows).
- **Cause:** the SQL prompt said "Day/month questions: prefer daily_summary…" and "use LIMIT
  (at most 500 rows)"; the chart guide also favours time series.
- **Fix (`agent/prompts.py`, WRITE_SQL_SYSTEM):** explicit "Result shape" rules (one total →
  one row; N periods → N rows; top N → ORDER BY + LIMIT N; series only for breakdowns/trends;
  "never return daily rows to answer a question about a month or a year"); source choice by
  granularity; LIMIT only for top-N or capping long series.

### 1.4 Wrong unit in the answer ("26,195,000 GWh")
- **Cause:** the SQL summed MWh into a column named `total_generation_gwh`; the explain step
  trusted the name.
- **Fix:** SQL prompt rule: the unit in the alias must match the conversion
  (`sum(generation_mwh) / 1e6 AS total_twh`, never a raw MWh sum named `_gwh`).

### 1.5 Over-clarification after fix 1.1
- **Symptom:** once the model was told not to refuse because of the period, it asked instead:
  "What was the average day-ahead price in March 2024?" → *"The whole month of March 2024 or a
  specific week?"*. Its own reason claimed the question "does not specify a period".
- **Fixes:**
  - Prompt: "A named period (a year, quarter, month, season, week or day) is complete and clear:
    never ask whether the user means the whole period or a part of it", with an example.
    On its own this did **not** help (targeted re-run).
  - Code (`apply_period_rules`): "clarify" is defined as *no time hint at all*, so if the model
    filled `periods`, the decision becomes "answer" (and still goes through the coverage check).
  - `periods` moved before `decision` in the schema so the model resolves the period first.
- **Still open:** when the model asks *and* leaves `periods` empty, the rule cannot fire
  (see 5.1).

### 1.6 "Most recent" season resolved to the earliest one
- **Symptom:** "What was the electricity price in winter?" → winter 2023/24 instead of 2025/26.
- **Fix:** prompt default extended to quarters and seasons, with season definitions and
  "(the latest one that has fully ended, not the earliest)". Partly effective: "Sommer" was
  still resolved to 2023 once in the after-run.

### 1.7 Self-correction barely helps (+1.4 pp)
- **Cause:** retries only trigger on SQL errors and the plausibility checks in
  `agent/checks.py`. The dominant mistakes (wrong period, wrong shape, UTC filtering) produce
  valid queries, so no retry happens.
- **Status:** not fixed (proposed: let the understand step state the expected result shape and
  flag mismatches in `checks.py`). See 5.3.

---

## 2. Evaluation pipeline errors (my own bugs and design mistakes)

### 2.1 API failures would have been scored as wrong answers
- **Cause:** `GridQueryAgent.ask()` catches every exception and returns status *failed*. A dead
  API key would have produced 186 "wrong answers" and saved them as done in the checkpoint.
- **Fix:** new field `AgentResult.error` ("ExceptionType: message"). The runner treats
  anything except `LLMOutputError` as an infrastructure error: not scored, not saved as a
  result (retried on resume), spend still recorded; the run stops after 3 in a row.

### 2.2 Spending cap impossible: cost unknown
- **Cause:** `.env` has no token prices, so `Usage.cost_usd` is `None`.
- **Fix:** the eval CLI refuses to start when prices are unknown; the runner raises
  `SpendingCapError` if a result has unknown cost. For the runs, prices were passed as env
  vars (gpt-4o-mini: $0.15 / $0.60 per million tokens). **To do:** add
  `GRIDQUERY_LLM_INPUT_COST_PER_MTOK` / `GRIDQUERY_LLM_OUTPUT_COST_PER_MTOK` to `.env`.

### 2.3 Scoring too strict: correct numbers marked wrong
- **Symptom (baseline):** misses with correct values:
  - `agg_03`: source shown as `name_en` ("Solar (photovoltaics)") vs gold key `solar`;
  - `cmp_07`: `year` labels vs gold `month_local`;
  - `cmp_04`: two years as two columns instead of two rows;
  - `agg_08`: weekdays numbered 0–6 (`strftime('%w')`) vs gold ISO weekday 1–7.
- **Fix (`evaluation/matching.py`):**
  - `compare_values` (primary metric): every gold number and every gold timestamp must appear
    in the result with one consistent unit scale; labels, column names, row order and layout
    are ignored; results longer than the gold still fail (except top-N prefixes), so daily
    dumps cannot pass by accident.
  - `compare_results` (strict metric) kept and reported next to it.
  - Source-name aliases (`name_en`/`name_de` → key) read from `energy_sources` via the
    safe executor.
- **Checked:** every case that flipped to correct was inspected; all had correct values.

### 2.4 Verdicts frozen into the checkpoint
- **Problem:** scoring was computed at run time and stored. Every gold correction would have
  invalidated saved answers (the fingerprint covered the whole case) and required a paid re-run.
- **Fix:** the checkpoint stores only what the agent did; `runner.judge()` scores at report
  time against the current eval set. The case fingerprint now covers only question + reference
  date. After editing gold answers, `--resume latest` re-scores without LLM calls.
- **Migration:** the existing baseline checkpoint's fingerprints were rewritten (questions
  unchanged; backup kept in the session scratchpad) and the run was re-scored at $0.

### 2.5 Matching bugs caught by tests
- Dates vs timestamps: `2024-03-01` did not match `2024-03-01T00:00:00` (dates went through
  the generic string path). Fixed by normalizing all ISO dates/timestamps the same way.
- Null and string comparisons depended on the unit scale. Fixed: scale applies to numbers only.
- `2025-01-20T17:00` (no seconds) did not match `2025-01-20T17:00:00`. Fixed by parsing with
  `datetime.fromisoformat` and emitting a canonical form instead of regex patching.

### 2.6 Smaller issues
| Problem | Fix |
|---|---|
| `UnicodeEncodeError` (cp1252) when printing German text on Windows | Run with `PYTHONIOENCODING=utf-8` |
| `data/*` in `.gitignore` hid `data/eval/eval_set.jsonl` | Added `!data/eval/`; checkpoints (`*.progress.jsonl`) ignored, reports kept |
| `from conftest import …` could resolve to the root `tests/conftest.py` | Helpers moved to `tests/evaluation/eval_helpers.py` (same pattern as `agent_helpers.py`) |
| ruff F821 (missing imports) after splitting the helpers | Imports added |
| Markdown report rows shorter than the header when the Δ column exists | Padding; rows built from a `VALUE_ROWS` table |
| Lines over 100 characters (ruff E501) in report and prompt | Reformatted / wrapped |
| Shell heredocs with backslashes failed ("unexpected EOF") or silently didn't match | Patches written as Python script files instead |
| One patch failed but the next command still ran (commands on separate lines, not `&&`) | Re-ran correctly; cost ~1 cent of wasted API calls |
| German refusal message said "Für 2015-01-01 **to** 2015-12-31" | `coverage.describe()` takes the language ("bis" in German) |
| A test I wrote hedged its own assertion instead of checking the exact German text | Replaced with an exact-string assertion |
| Node tests built state by hand without the new `data_range` (KeyError) | Tests updated; `load_context` provides `data_range` |

---

## 3. Evaluation set (gold data) issues

| Case | Problem | Fix |
|---|---|---|
| "Summer 2023 vs summer 2024" (your example) | Data starts 2023-09-25, summer 2023 not covered | Made a refusal case (`refuse_10`); added answerable summer 2024 vs 2025 (`cmp_01`) |
| "Most electricity in 2023" (your example) | 2023 only partly covered | Kept answerable, note says the answer must mention partial coverage |
| `lookup_16` nuclear 2024 | Series is all zeros and ends 2024-01-30 | Gold = 0; `required_columns` excludes the informational month columns |
| `lookup_05`, `de_09` | Gold showed top 3 rows to expose ties; a correct single-row answer would fail | No ties found → gold changed to `LIMIT 1` |
| `agg_02` | Informational `first_month` column would fail correct answers | `required_columns = [source, generation_twh]` |
| `agg_08` | `iso_weekday` was only a sort helper | `required_columns = [weekday, avg_daily_consumption_gwh]` |
| `amb_04/05/06/10` (winter, Q1, May, Sommer) | Labelled ambiguous, but the prompt default ("most recent") deliberately answers them | Your decision: keep the default → now answerable, category `implicit_period`, gold = most recent complete period; ids kept |
| Relative dates ("last year", "yesterday") | Depend on the run date | Every case has `reference_date: 2026-10-06`; the runner pins the agent's "today" to it |

---

## 4. Configuration / environment

| Problem | Status |
|---|---|
| `.env` sets `GRIDQUERY_MAX_SQL_RETRIES`, which no code reads (the setting is `GRIDQUERY_MAX_SQL_ATTEMPTS`) | Open; harmless because the default is 3 |
| No LLM prices in `.env` | Open; add them (see 2.2) |

---

## 5. Still open

### 5.1 Refusals and clarifications the code cannot override
- The model still sometimes refuses with "no data for 2024" (`agg_04`, `agg_12`, `calc_04`,
  `calc_06`). The code cannot tell a period-based refusal from a topic-based one (e.g. CO2 in
  2024). Idea: add a field for *why* it is unanswerable (topic vs. other) and only trust topic
  refusals.
- It sometimes asks "whole year or a month?" **without** filling `periods`, so neither code
  rule fires. `refuse_07` (2015) regressed this way: it should be refused by the coverage check.

### 5.2 SQL mistakes
- Filtering on UTC instead of German time, off by 1–2 hours (`agg_02`, `cmp_02`, `de_11`,
  baseline `cmp_07`). The prompt already has the rule; a worked example may help.
- `AND`/`OR` without parentheses (`amb_06`: `month = … AND source = 'a' OR source = 'b'`
  → 86 TWh of wind in May).
- Comparisons still returning too many rows (`cmp_01`, `cmp_04`, `cmp_11`).

### 5.3 Self-correction has nothing to correct
Add semantic checks (expected result shape from the understand step, empty result for a
covered period, unit/magnitude sanity) so retries fire on the mistakes in 5.2.

### 5.4 Remaining clarification design
- Missing *measure* ("Show me the trend", "How high was the peak?") is not covered by the clarify
  rule, which only talks about missing periods (item 4 of the original list).
- `amb_09` ("expensive last weekend?") is now refused by code (3–4 Oct 2026 is after the data).
  Arguably correct; the eval still expects a clarification. Gold decision.

### 5.5 Scoring
- `cmp_04`: correct values in a one-column-per-year layout fail because the gold `year` labels
  count as required numbers. Setting `required_columns` fixes it but loses the check of which
  value belongs to which year. Gold decision.
- `calc_06` (correlation): the SQL prompt asks for data points plus `corr() OVER ()`, so the
  result has many rows while the gold has one; it would fail even when correct.

### 5.6 Measurement noise
At temperature 0 some cases still flip between runs (e.g. `amb_01`). With 73 answerable
questions one case is 1.4 pp, so differences of 2–3 pp between runs are noise.

---

## 6. API, UI and Docker

| Problem | Fix / status |
|---|---|
| One agent shares one DuckDB connection between its steps; concurrent requests would share it | `agent/pool.py`: a pool of independent agents (`GRIDQUERY_API_MAX_CONCURRENCY`), each with its own read-only connection |
| A missing database or API key would crash the server at startup | The app starts anyway; `/health` returns 503 with the reason, `/ask` returns 503 |
| The agent appends the SQL to its answer, which duplicates the `sql` field | `AgentResult.answer_text` strips exactly that block; API and UI use it |
| UI examples that the agent got wrong: renewable share 2025 (averaged monthly shares: 60.72% instead of 60.2%), price change 2024→2025 (UTC filter added a 2026 row), highest-consumption day (explanation said "the only day with data") | Replaced by examples verified against the real agent; the agent issues stay open (see 5.2) |
| `docker compose config` printed the real OpenAI and LangSmith keys from `.env` into the session transcript | Not written anywhere else. **Rotate both keys.** Validate with `docker compose config --no-interpolate` |
| Docker Desktop was not running, so the image was **not built or run** in this session | Open: run `docker compose up --build` once and check http://localhost:8000 |
| Chrome extension not connected, so no visual check of the UI | Checked instead via `/ui/config` and by running the UI handler with the real agent |

---

## 7. Spend

| Item | Cost |
|---|---|
| Smoke test (3 cases) | $0.003 |
| Baseline full run | $0.152 |
| Targeted checks (4 runs, one wasted by a failed patch) | ~$0.04 |
| After-fix full run | $0.190 |
| **Total** | **~$0.39** |
