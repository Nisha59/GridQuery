"""Rule-based plausibility checks on query results.

These run after every query. Each problem found becomes feedback for the next
SQL attempt. They are deliberately simple rules on column names and values:
cheap, predictable, and easy to test.
"""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from gridquery.agent.schemas import Attempt
from gridquery.tools.sql_executor import QueryResult

MIN_PRICE = -500.0
MAX_PRICE = 4_000.0


def normalize_sql(sql: str) -> str:
    """Normalize SQL text for comparing attempts (whitespace, case, trailing semicolon)."""
    return " ".join(sql.split()).rstrip(";").strip().lower()


def check_result(result: QueryResult, previous_attempts: Sequence[Attempt] = ()) -> list[str]:
    """Return problems with a query result; an empty list means it looks fine.

    Args:
        result: The executed query.
        previous_attempts: Earlier attempts for the same question.
    """
    if not result.ok:
        return [f"{result.error_kind} error: {result.error}"]
    if result.row_count == 0:
        return [] if _empty_confirmed(result, previous_attempts) else [_EMPTY_MESSAGE]

    issues = []
    if all(value is None for row in result.rows for value in row):
        issues.append("All values are NULL. Check the filters, joins and date range.")
    for column_index, column in enumerate(result.columns):
        values = _numbers(row[column_index] for row in result.rows)
        issues += _check_column(column.lower(), values)
    if result.truncated:
        issues.append(
            f"The result has more than {result.row_count} rows and was cut off. "
            "Aggregate the data or add a LIMIT."
        )
    return issues


_EMPTY_MESSAGE = (
    "The query returned no rows. Check the date range (data coverage), filters and source "
    "keys. If an empty result is really the correct answer, return exactly the same SQL again."
)


def _empty_confirmed(result: QueryResult, previous_attempts: Sequence[Attempt]) -> bool:
    """An empty result is accepted if the model re-submits the same empty query."""
    return any(
        attempt.ok
        and attempt.row_count == 0
        and normalize_sql(attempt.sql) == normalize_sql(result.sql)
        for attempt in previous_attempts
    )


SHARE_WORDS = frozenset({"share", "fraction", "ratio"})
PERCENT_WORDS = frozenset({"pct", "percent"})
ENERGY_WORDS = frozenset({"mwh", "gwh", "twh", "generation", "consumption"})


def _check_column(column: str, values: list[float]) -> list[str]:
    """Plausibility rules based on the words in a snake_case column name."""
    if not values:
        return []
    words = set(column.split("_"))
    low, high = min(values), max(values)
    if words & SHARE_WORDS:
        limit = 100.0 if words & PERCENT_WORDS else 1.0
        if low < 0 or high > limit:
            return [f"Column {column} has values from {low:g} to {high:g}, outside 0-{limit:g}."]
    elif "price" in words:
        if low < MIN_PRICE or high > MAX_PRICE:
            return [
                f"Column {column} has prices from {low:g} to {high:g} EUR/MWh, outside the "
                f"market limits {MIN_PRICE:g} to {MAX_PRICE:g}. Check units and aggregation."
            ]
    elif words & ENERGY_WORDS and low < 0:
        return [f"Column {column} has negative energy values (min {low:g})."]
    return []


def _numbers(values: Any) -> list[float]:
    """The numeric values of a column (DuckDB returns DECIMAL columns as Decimal)."""
    return [
        float(v) for v in values if isinstance(v, int | float | Decimal) and not isinstance(v, bool)
    ]
