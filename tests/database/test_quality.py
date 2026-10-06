from datetime import datetime, timedelta

import duckdb

from gridquery.database.quality import (
    Severity,
    check_duplicates,
    check_missing_hours,
    check_negative_values,
    check_price_limits,
    has_errors,
    run_checks,
)

T0 = datetime(2026, 1, 1)
HOUR = timedelta(hours=1)


def insert_consumption(con: duckdb.DuckDBPyConnection, hours: list[int], value: float = 1.0):
    con.executemany(
        "INSERT INTO consumption VALUES (?, ?)", [(T0 + h * HOUR, value) for h in hours]
    )


def by_series(results):
    return {r.series: r for r in results}


def test_all_checks_pass_on_clean_data(con: duckdb.DuckDBPyConnection) -> None:
    insert_consumption(con, [0, 1, 2])
    con.execute("INSERT INTO day_ahead_prices VALUES (?, -120.5)", [T0])

    results = run_checks(con)

    assert all(r.passed for r in results)
    assert not has_errors(results)


def test_missing_hours_are_counted_with_examples(con: duckdb.DuckDBPyConnection) -> None:
    insert_consumption(con, [0, 1, 4, 5])

    result = by_series(check_missing_hours(con))["consumption"]

    assert result.issue_count == 2
    assert result.examples == ["2026-01-01 02:00 UTC", "2026-01-01 03:00 UTC"]
    assert result.severity is Severity.WARNING


def test_utc_series_across_dst_switch_has_no_missing_hours(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # 2026-03-29: Berlin clocks jump 02:00 -> 03:00, but UTC hours stay continuous.
    start = datetime(2026, 3, 28, 23)
    con.executemany(
        "INSERT INTO consumption VALUES (?, 1.0)", [(start + h * HOUR,) for h in range(23)]
    )

    assert by_series(check_missing_hours(con))["consumption"].passed
    hours = con.execute(
        "SELECT hours FROM daily_summary WHERE date_local = DATE '2026-03-29'"
    ).fetchone()
    assert hours == (23,)


def test_negative_generation_is_an_error_but_negative_prices_are_fine(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.execute("INSERT INTO generation VALUES (?, 'solar', -3.0), (?, 'lignite', 10.0)", [T0, T0])
    con.execute("INSERT INTO day_ahead_prices VALUES (?, -50.0)", [T0])

    results = by_series(check_negative_values(con))

    assert results["solar"].issue_count == 1
    assert results["solar"].severity is Severity.ERROR
    assert results["lignite"].passed
    assert "day_ahead_price" not in results
    assert has_errors(list(results.values()))


def test_duplicates_are_detected() -> None:
    # The real schema's primary keys make duplicates impossible, so use bare tables
    # with the same columns to prove the check itself works.
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE generation (timestamp_utc TIMESTAMP, source VARCHAR, generation_mwh DOUBLE)"
    )
    con.execute("CREATE TABLE consumption (timestamp_utc TIMESTAMP, consumption_mwh DOUBLE)")
    con.execute("CREATE TABLE day_ahead_prices (timestamp_utc TIMESTAMP, price_eur_per_mwh DOUBLE)")
    con.execute("INSERT INTO consumption VALUES (?, 1), (?, 2), (?, 3)", [T0, T0, T0 + HOUR])

    result = by_series(check_duplicates(con))["consumption"]

    assert result.issue_count == 1
    assert result.examples == ["2026-01-01 00:00 UTC"]


def test_primary_key_blocks_duplicates(con: duckdb.DuckDBPyConnection) -> None:
    insert_consumption(con, [0])
    try:
        insert_consumption(con, [0])
    except duckdb.ConstraintException:
        pass
    else:
        raise AssertionError("duplicate hour was accepted")


def test_prices_outside_market_limits_are_flagged(con: duckdb.DuckDBPyConnection) -> None:
    con.executemany(
        "INSERT INTO day_ahead_prices VALUES (?, ?)",
        [(T0, -500.0), (T0 + HOUR, 4000.0), (T0 + 2 * HOUR, 5000.0)],
    )

    [result] = check_price_limits(con)

    assert result.issue_count == 1
    assert result.examples == ["2026-01-01 02:00 UTC"]
