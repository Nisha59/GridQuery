from datetime import date, datetime, timedelta

import duckdb
import pytest

HOUR = timedelta(hours=1)


def test_to_berlin_time_handles_winter_and_summer(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT to_berlin_time(TIMESTAMP '2026-01-15 12:00'), "
        "to_berlin_time(TIMESTAMP '2026-07-15 12:00')"
    ).fetchone()
    assert rows == (datetime(2026, 1, 15, 13), datetime(2026, 7, 15, 14))


def test_daily_views_group_by_german_date(con: duckdb.DuckDBPyConnection) -> None:
    # 23:00 UTC on Jan 1 is already Jan 2 in Berlin.
    con.execute(
        "INSERT INTO generation VALUES "
        "(TIMESTAMP '2026-01-01 22:00', 'solar', 1), (TIMESTAMP '2026-01-01 23:00', 'solar', 2)"
    )

    rows = con.execute(
        "SELECT date_local, generation_mwh FROM generation_daily ORDER BY 1"
    ).fetchall()

    assert rows == [(date(2026, 1, 1), 1.0), (date(2026, 1, 2), 2.0)]


def test_summaries_compute_renewable_share_and_prices(con: duckdb.DuckDBPyConnection) -> None:
    t0 = datetime(2026, 5, 31, 22)  # 2026-06-01 00:00 Berlin
    con.executemany(
        "INSERT INTO generation VALUES (?, ?, ?)",
        [(t0, "solar", 30.0), (t0, "lignite", 70.0), (t0 + HOUR, "wind_onshore", 100.0)],
    )
    con.executemany("INSERT INTO consumption VALUES (?, ?)", [(t0, 90.0), (t0 + HOUR, 110.0)])
    con.executemany("INSERT INTO day_ahead_prices VALUES (?, ?)", [(t0, -10.0), (t0 + HOUR, 50.0)])

    hourly = con.execute(
        "SELECT renewable_share_of_generation FROM hourly_summary ORDER BY timestamp_utc"
    ).fetchall()
    daily = con.execute(
        "SELECT date_local, total_generation_mwh, renewable_share_of_generation, consumption_mwh, "
        "avg_price_eur_per_mwh, negative_price_hours, hours FROM daily_summary"
    ).fetchall()
    month = con.execute("SELECT month_local, fossil_generation_mwh FROM monthly_summary").fetchall()

    assert hourly == [(0.3,), (1.0,)]
    assert daily == [(date(2026, 6, 1), 200.0, 0.65, 200.0, 20.0, 1, 2)]
    assert month == [(date(2026, 6, 1), 70.0)]


def test_data_coverage_lists_each_series(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSERT INTO consumption VALUES (TIMESTAMP '2026-01-01 00:00', 1)")
    con.execute("INSERT INTO consumption VALUES (TIMESTAMP '2026-01-01 05:00', 1)")

    rows = con.execute(
        "SELECT series, first_timestamp_utc, last_timestamp_utc, row_count FROM data_coverage"
    ).fetchall()

    # Series without data are still listed, with row_count 0, so gaps are visible.
    assert rows == [
        ("consumption", datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 5), 2),
        ("day_ahead_price", None, None, 0),
    ]


def test_generation_requires_known_source(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO generation VALUES (TIMESTAMP '2026-01-01', 'coal', 1)")


def test_tables_and_columns_are_documented(con: duckdb.DuckDBPyConnection) -> None:
    undocumented = con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE comment IS NULL "
        "UNION ALL SELECT view_name FROM duckdb_views() WHERE NOT internal AND comment IS NULL"
    ).fetchall()
    assert undocumented == []
