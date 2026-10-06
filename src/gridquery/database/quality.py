"""Data quality checks on the loaded database.

Each check is a SQL query that returns the problems it finds, so the same checks
can run right after a load or at any time against a read-only connection.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import duckdb

logger = logging.getLogger(__name__)

# Harmonised min/max clearing prices of the European day-ahead market (SDAC).
MIN_DAY_AHEAD_PRICE = -500.0
MAX_DAY_AHEAD_PRICE = 4000.0
MAX_EXAMPLES = 3

# Every hourly series as (series, timestamp_utc, value), so checks can treat them alike.
ALL_SERIES_SQL = """
    SELECT source AS series, timestamp_utc, generation_mwh AS value FROM generation
    UNION ALL SELECT 'consumption', timestamp_utc, consumption_mwh FROM consumption
    UNION ALL SELECT 'day_ahead_price', timestamp_utc, price_eur_per_mwh FROM day_ahead_prices
"""


class Severity(StrEnum):
    """How serious a failed check is."""

    ERROR = "error"  # Data is wrong; results would be misleading.
    WARNING = "warning"  # Data is incomplete or unusual; worth knowing about.


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check for one series."""

    check: str
    series: str
    severity: Severity
    issue_count: int
    examples: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if the check found no issues."""
        return self.issue_count == 0


def check_missing_hours(con: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Find hours without a value between each series' first and last hour.

    UTC has no daylight-saving jumps, so every series should be a gap-free hourly sequence.
    """
    rows = con.execute(f"""
        WITH series AS ({ALL_SERIES_SQL}),
        bounds AS (
            SELECT series, min(timestamp_utc) AS first_hour, max(timestamp_utc) AS last_hour
            FROM series GROUP BY series
        ),
        expected AS (
            SELECT b.series, h.generate_series AS timestamp_utc
            FROM bounds AS b,
                 LATERAL generate_series(b.first_hour, b.last_hour, INTERVAL 1 HOUR) AS h
        ),
        missing AS (
            SELECT e.series, e.timestamp_utc
            FROM expected AS e
            ANTI JOIN series AS s USING (series, timestamp_utc)
        )
        SELECT b.series, count(m.timestamp_utc),
               list(m.timestamp_utc ORDER BY m.timestamp_utc)
                   FILTER (WHERE m.timestamp_utc IS NOT NULL)
        FROM bounds AS b LEFT JOIN missing AS m USING (series)
        GROUP BY b.series ORDER BY b.series
    """).fetchall()
    return [
        CheckResult("missing_hours", series, Severity.WARNING, count, _examples(hours))
        for series, count, hours in rows
    ]


def check_negative_values(con: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Find negative generation or consumption, which is physically impossible.

    Prices are excluded: negative day-ahead prices are real and common.
    """
    rows = con.execute(f"""
        WITH series AS ({ALL_SERIES_SQL})
        SELECT series,
               count(*) FILTER (WHERE value < 0),
               list(timestamp_utc ORDER BY timestamp_utc) FILTER (WHERE value < 0)
        FROM series WHERE series <> 'day_ahead_price'
        GROUP BY series ORDER BY series
    """).fetchall()
    return [
        CheckResult("negative_values", series, Severity.ERROR, count, _examples(hours))
        for series, count, hours in rows
    ]


def check_duplicates(con: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Find hours that appear more than once in a series.

    The primary keys already prevent this; the check guards the invariant anyway.
    """
    rows = con.execute(f"""
        WITH series AS ({ALL_SERIES_SQL}),
        repeated AS (
            SELECT series, timestamp_utc FROM series
            GROUP BY series, timestamp_utc HAVING count(*) > 1
        )
        SELECT s.series, count(r.timestamp_utc),
               list(r.timestamp_utc ORDER BY r.timestamp_utc)
                   FILTER (WHERE r.timestamp_utc IS NOT NULL)
        FROM (SELECT DISTINCT series FROM series) AS s LEFT JOIN repeated AS r USING (series)
        GROUP BY s.series ORDER BY s.series
    """).fetchall()
    return [
        CheckResult("duplicate_hours", series, Severity.ERROR, count, _examples(hours))
        for series, count, hours in rows
    ]


def check_price_limits(con: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Find day-ahead prices outside the market's technical price limits."""
    count, hours = con.execute(
        """
        SELECT count(*), list(timestamp_utc ORDER BY timestamp_utc)
        FROM day_ahead_prices WHERE price_eur_per_mwh NOT BETWEEN ? AND ?
        """,
        [MIN_DAY_AHEAD_PRICE, MAX_DAY_AHEAD_PRICE],
    ).fetchone()
    return [CheckResult("price_limits", "day_ahead_price", Severity.ERROR, count, _examples(hours))]


CHECKS: tuple[Callable[[duckdb.DuckDBPyConnection], list[CheckResult]], ...] = (
    check_missing_hours,
    check_negative_values,
    check_duplicates,
    check_price_limits,
)


def run_checks(con: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Run all checks and log each result."""
    results = [result for check in CHECKS for result in check(con)]
    for result in results:
        log_result(result)
    failed = [r for r in results if not r.passed]
    logger.info(
        "Quality checks: %d passed, %d warnings, %d errors",
        len(results) - len(failed),
        sum(r.severity is Severity.WARNING for r in failed),
        sum(r.severity is Severity.ERROR for r in failed),
    )
    return results


def has_errors(results: list[CheckResult]) -> bool:
    """True if any failed check has ERROR severity."""
    return any(not r.passed and r.severity is Severity.ERROR for r in results)


def log_result(result: CheckResult) -> None:
    """Log one check result: passes at DEBUG, failures at WARNING or ERROR."""
    if result.passed:
        logger.debug("PASS %s [%s]", result.check, result.series)
        return
    level = logging.ERROR if result.severity is Severity.ERROR else logging.WARNING
    logger.log(
        level,
        "FAIL %s [%s]: %d issues, e.g. %s",
        result.check,
        result.series,
        result.issue_count,
        ", ".join(result.examples),
    )


def _examples(timestamps: list | None) -> list[str]:
    """Format the first few offending timestamps for logs."""
    return [f"{ts:%Y-%m-%d %H:%M} UTC" for ts in (timestamps or [])[:MAX_EXAMPLES]]
