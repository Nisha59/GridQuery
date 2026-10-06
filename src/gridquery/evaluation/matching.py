"""Compare an agent's query result with the gold result (execution accuracy).

Two levels, both lenient about presentation and strict about values:

* compare_values (primary): every gold number, and every gold timestamp (the
  answer to "when?"), appears in the agent's result. Labels, column names,
  row order and layout (one row per year vs one column per year) are ignored.
* compare_results (strict): additionally, every required gold column must be
  paired with a distinct agent column such that whole rows agree.

Shared rules:

* Results must not be longer than the gold result, unless the gold query has a
  LIMIT (a "top N" question): then the agent's first N rows are compared. This
  keeps a dump of daily rows from passing for a yearly total.
* Numbers match within the precision the gold value is written with (64.7 means
  +/-0.05) or 0.01 %, whichever is larger, after one unit scale for the whole
  column (strict) or result (values): MWh/GWh/TWh, fraction/percent.
* Strings are compared case-insensitively with "_"/"-" treated as spaces;
  `aliases` maps display names (e.g. "Solar (photovoltaics)") to keys ("solar").
  Timestamps ignore a "T" separator, a time-zone offset and a midnight time.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

# Allowed unit conversions agent = gold * scale (e.g. TWh -> MWh, percent -> fraction).
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e2, 1e-2)
RELATIVE_TOLERANCE = 1e-4
MAX_DECIMALS = 6

_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)
_TZ_OFFSET = re.compile(r"(?:[+-]\d{2}:?\d{2}|Z)$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")

Value = tuple[str, Any]  # ("num", float) | ("str", str) | ("null", None)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of comparing one agent result with the gold result.

    Attributes:
        match: True if all required gold columns and rows were found.
        reason: Short explanation, mainly for mismatches.
        column_map: Gold column -> agent column for the paired columns.
    """

    match: bool
    reason: str
    column_map: dict[str, str] = field(default_factory=dict)


def compare_results(
    gold_columns: list[str],
    gold_rows: list[list[Any]],
    agent_columns: list[str],
    agent_rows: list[list[Any]],
    *,
    required_columns: list[str] | None = None,
    gold_sql: str | None = None,
    aliases: Mapping[str, str] | None = None,
) -> MatchResult:
    """Strict check: the agent's rows contain the gold result column by column.

    Args:
        gold_columns: Column names of the gold result.
        gold_rows: Gold rows (JSON values).
        agent_columns: Column names returned by the agent.
        agent_rows: Rows returned by the agent (JSON values).
        required_columns: Gold columns that must be matched (default: all).
        gold_sql: Gold query; a LIMIT in it allows comparing a prefix of longer results.
        aliases: Normalized string -> canonical string (e.g. source display names).

    Returns:
        The match decision with a reason and the column pairing.
    """
    rows, problem = _comparable_rows(gold_rows, agent_rows, gold_sql, allow_fewer=False)
    if problem:
        return MatchResult(False, problem)

    required = required_columns or gold_columns
    gold_values = {
        c: [_normalize(r[gold_columns.index(c)], aliases) for r in gold_rows] for c in required
    }
    agent_values = [[_normalize(r[i], aliases) for r in rows] for i in range(len(agent_columns))]

    column_map: dict[str, str] = {}
    scales: dict[str, float] = {}
    used: set[int] = set()
    for column in required:
        found = _find_column(gold_values[column], agent_values, used)
        if found is None:
            return MatchResult(False, f"no agent column matches gold column {column!r}", column_map)
        index, scale = found
        used.add(index)
        column_map[column] = agent_columns[index]
        scales[column] = scale

    if not _rows_match(required, gold_values, agent_values, column_map, agent_columns, scales):
        return MatchResult(False, "columns match individually but rows do not line up", column_map)
    return MatchResult(True, "match", column_map)


def compare_values(
    gold_columns: list[str],
    gold_rows: list[list[Any]],
    agent_rows: list[list[Any]],
    *,
    required_columns: list[str] | None = None,
    gold_sql: str | None = None,
    aliases: Mapping[str, str] | None = None,
) -> MatchResult:
    """Value check: every gold number and timestamp appears in the agent's result.

    Arguments as for compare_results. Gold dates and labels are not required (the
    question names them; the agent may label rows differently), but timestamps are.
    """
    rows, problem = _comparable_rows(gold_rows, agent_rows, gold_sql, allow_fewer=True)
    if problem:
        return MatchResult(False, problem)

    indexes = [gold_columns.index(c) for c in (required_columns or gold_columns)]
    gold_cells = [_normalize(row[i], aliases) for row in gold_rows for i in indexes]
    agent_cells = [_normalize(value, aliases) for row in rows for value in row]

    timestamps = [c for c in gold_cells if c[0] == "str" and _is_timestamp(c[1])]
    missing = [c[1] for c in timestamps if c not in agent_cells]
    if missing:
        return MatchResult(False, f"timestamp {missing[0]} not in result")
    numbers = [c for c in gold_cells if c[0] == "num"]
    candidates = [c for c in agent_cells if c[0] == "num"]
    if any(_all_found(numbers, candidates, scale) for scale in SCALES):
        return MatchResult(True, "values match")
    unmatched = next(
        (g[1] for g in numbers if not any(_equal(g, a, s) for a in candidates for s in SCALES)),
        None,
    )
    if unmatched is None:
        return MatchResult(False, "values found only at different unit scales")
    return MatchResult(False, f"value {unmatched:g} not in result")


def has_limit(sql: str) -> bool:
    """True if the query has a LIMIT clause (a "top N" question)."""
    return bool(_LIMIT.search(sql))


def _comparable_rows(
    gold_rows: list[list[Any]],
    agent_rows: list[list[Any]],
    gold_sql: str | None,
    *,
    allow_fewer: bool,
) -> tuple[list[list[Any]], str | None]:
    """Agent rows to compare (a prefix for top-N gold queries), or a reason to reject."""
    if not agent_rows:
        return agent_rows, "agent returned no rows"
    if len(agent_rows) == len(gold_rows) or (allow_fewer and len(agent_rows) < len(gold_rows)):
        return agent_rows, None
    if gold_sql and has_limit(gold_sql) and len(agent_rows) > len(gold_rows):
        return agent_rows[: len(gold_rows)], None
    return agent_rows, f"row count {len(agent_rows)} != expected {len(gold_rows)}"


def _all_found(gold: list[Value], candidates: list[Value], scale: float) -> bool:
    """Pair every gold number with a distinct candidate cell at this scale."""
    remaining = list(candidates)
    for value in gold:
        partner = next((i for i, c in enumerate(remaining) if _equal(value, c, scale)), None)
        if partner is None:
            return False
        remaining.pop(partner)
    return True


def _is_timestamp(text: str) -> bool:
    """True for canonical 'YYYY-MM-DD HH:MM:SS' strings (dates have no time part)."""
    return bool(_ISO_DATE.match(text)) and " " in text


def _find_column(
    gold: list[Value], agent_columns: list[list[Value]], used: set[int]
) -> tuple[int, float] | None:
    """First unused agent column whose sorted values equal the gold column's, and its scale."""
    gold_sorted = sorted(gold, key=_sort_key)
    for index, values in enumerate(agent_columns):
        if index in used:
            continue
        values_sorted = sorted(values, key=_sort_key)
        for scale in SCALES:
            if all(_equal(g, a, scale) for g, a in zip(gold_sorted, values_sorted, strict=True)):
                return index, scale
    return None


def _rows_match(
    required: list[str],
    gold_values: dict[str, list[Value]],
    agent_values: list[list[Value]],
    column_map: dict[str, str],
    agent_columns: list[str],
    scales: dict[str, float],
) -> bool:
    """Pair every gold row with a distinct agent row that agrees on all paired columns."""
    indexes = [agent_columns.index(column_map[c]) for c in required]
    remaining = set(range(len(agent_values[0]) if agent_values else 0))
    for row in range(len(gold_values[required[0]])):
        partner = next(
            (
                candidate
                for candidate in sorted(remaining)
                if all(
                    _equal(gold_values[c][row], agent_values[i][candidate], scales[c])
                    for c, i in zip(required, indexes, strict=True)
                )
            ),
            None,
        )
        if partner is None:
            return False
        remaining.discard(partner)
    return True


def normalize_text(text: str) -> str:
    """Lower-case, "_"/"-" as spaces, single spaces: how strings are compared."""
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _normalize(value: Any, aliases: Mapping[str, str] | None = None) -> Value:
    """Map a JSON value to a comparable ("num" | "str" | "null", value) pair."""
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("str", str(value).lower())
    if isinstance(value, int | float):
        return ("num", float(value))
    text = str(value).strip()
    try:
        return ("num", float(text))
    except ValueError:
        pass
    if _ISO_DATE.match(text):
        return ("str", _canonical_datetime(text))
    text = normalize_text(text)
    return ("str", (aliases or {}).get(text, text))


def _canonical_datetime(text: str) -> str:
    """'YYYY-MM-DD' for midnight, else 'YYYY-MM-DD HH:MM:SS'; the local wall time is kept."""
    try:
        parsed = datetime.fromisoformat(_TZ_OFFSET.sub("", text)).replace(tzinfo=None)
    except ValueError:
        return text
    if parsed.time() == time(0):
        return parsed.date().isoformat()
    return parsed.isoformat(sep=" ")


def _sort_key(value: Value) -> tuple[str, Any]:
    """Sort numbers numerically and strings alphabetically, never mixing types."""
    kind, content = value
    return (kind, content if content is not None else "")


def _equal(gold: Value, agent: Value, scale: float) -> bool:
    """Compare two normalized values; numbers within the gold value's precision."""
    if gold[0] != agent[0]:
        return False
    if gold[0] != "num":
        return gold[1] == agent[1]
    expected, actual = gold[1], agent[1] / scale
    if math.isnan(expected) or math.isnan(actual):
        return math.isnan(expected) and math.isnan(actual)
    tolerance = max(0.5 * 10 ** -_decimals(expected), RELATIVE_TOLERANCE * abs(expected))
    return abs(actual - expected) <= tolerance * (1 + 1e-9)


def _decimals(value: float) -> int:
    """Number of decimals the gold value is written with (capped)."""
    text = repr(value)
    if "e" in text or "." not in text:
        return MAX_DECIMALS if "e-" in text else 0
    fraction = text.split(".")[1].rstrip("0")
    return min(len(fraction), MAX_DECIMALS)
