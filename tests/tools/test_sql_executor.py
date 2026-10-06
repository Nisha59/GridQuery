import json
from pathlib import Path

import duckdb
import pytest

from gridquery.database.connection import connect
from gridquery.tools.sql_executor import ErrorKind, SafeSqlExecutor, agent_connection_config

# --- Structured results ----------------------------------------------------------------------


def test_successful_query_returns_structured_result(executor: SafeSqlExecutor) -> None:
    result = executor.run(
        "SELECT source, sum(generation_mwh) AS total FROM generation "
        "GROUP BY source ORDER BY source"
    )

    assert result.ok
    assert result.columns == ["source", "total"]
    assert result.rows == [("lignite", 4800.0), ("solar", 4800.0)]
    assert result.row_count == 2
    assert not result.truncated
    assert result.execution_time_ms > 0
    assert result.error is None and result.error_kind is None


def test_result_is_json_serializable(executor: SafeSqlExecutor) -> None:
    result = executor.run(
        "SELECT timestamp_utc, CAST(timestamp_utc AS DATE) AS d, 1.5::DECIMAL(4,2) AS x, "
        "INTERVAL 1 HOUR AS i FROM consumption LIMIT 1"
    )

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["ok"] is True
    assert payload["rows"] == [["2026-01-01T00:00:00", "2026-01-01", 1.5, 3600.0]]


def test_views_and_macro_work(executor: SafeSqlExecutor) -> None:
    result = executor.run(
        "SELECT to_berlin_time(min(timestamp_utc)) AS first_local, count(*) FROM hourly_summary"
    )
    assert result.ok, result.error
    assert result.rows[0][1] == 48


def test_allowed_tables_come_from_the_database(executor: SafeSqlExecutor) -> None:
    assert {"generation", "consumption", "day_ahead_prices", "daily_summary"} <= (
        executor.allowed_tables
    )


# --- Limits ----------------------------------------------------------------------------------


def test_results_are_capped_at_max_rows(db_path: Path) -> None:
    with SafeSqlExecutor(db_path, max_rows=10) as executor:
        result = executor.run("SELECT * FROM consumption")

    assert result.ok
    assert result.row_count == 10
    assert result.truncated


def test_exactly_max_rows_is_not_truncated(db_path: Path) -> None:
    with SafeSqlExecutor(db_path, max_rows=48) as executor:
        result = executor.run("SELECT * FROM consumption")
    assert (result.row_count, result.truncated) == (48, False)


def test_slow_queries_time_out(db_path: Path) -> None:
    with SafeSqlExecutor(db_path, timeout_seconds=0.2) as executor:
        result = executor.run(
            "SELECT sum(a.range * b.range) FROM range(1000000) a, range(1000000) b"
        )
        after = executor.run("SELECT 1 AS still_alive")

    assert result.error_kind is ErrorKind.TIMEOUT
    assert "timed out" in result.error
    assert result.execution_time_ms < 5_000
    assert after.ok  # the executor stays usable after an interrupt


# --- Errors come back as data, not exceptions ------------------------------------------------


def test_execution_errors_are_returned_for_self_repair(executor: SafeSqlExecutor) -> None:
    result = executor.run("SELECT no_such_column FROM consumption")

    assert result.error_kind is ErrorKind.EXECUTION
    assert "no_such_column" in result.error
    assert result.rows == [] and result.columns == []


@pytest.mark.parametrize(
    ("sql", "fragment"),
    [
        ("DROP TABLE consumption", "DROP"),
        ("DELETE FROM consumption", "DELETE"),
        ("INSERT INTO consumption VALUES ('2030-01-01', 1)", "INSERT"),
        ("UPDATE consumption SET consumption_mwh = 0", "UPDATE"),
        ("SELECT 1; DROP TABLE consumption", "one SQL statement"),
        ("SELECT * FROM read_csv('secret.csv')", "read_csv"),
        ("SELECT * FROM 'secret.csv'", "Unknown table"),
        ("ATTACH 'x.duckdb' AS x", "ATTACH"),
        ("COPY consumption TO 'dump.csv'", "COPY"),
        ("SET enable_external_access = true", "SET"),
        ("CREATE TEMP TABLE t AS SELECT 1", "CREATE"),
        ("SELECT * FROM query('DELETE FROM consumption')", "query"),
    ],
)
def test_dangerous_queries_are_rejected_and_data_is_unchanged(
    executor: SafeSqlExecutor, sql: str, fragment: str
) -> None:
    result = executor.run(sql)

    assert result.error_kind is ErrorKind.VALIDATION
    assert fragment.lower() in result.error.lower()
    assert executor.run("SELECT count(*) FROM consumption").rows == [(48,)]
    assert executor.run("SELECT count(*) FROM generation").rows == [(96,)]


def test_dangerous_queries_leave_no_files_behind(
    executor: SafeSqlExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for sql in [
        "COPY consumption TO 'dump.csv'",
        "COPY (SELECT 1) TO 'x.parquet' (FORMAT parquet)",
        "EXPORT DATABASE 'dump'",
        "ATTACH 'new.duckdb' AS n",
    ]:
        assert not executor.run(sql).ok
    assert list(tmp_path.iterdir()) == []


# --- Defense in depth: the connection blocks attacks even without the validator ---------------


@pytest.fixture
def raw_agent_connection(db_path: Path):
    """The agent's hardened connection, used directly to bypass the validator."""
    con = connect(db_path, read_only=True, config=agent_connection_config("1GB"))
    yield con
    con.close()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM consumption",
        "DROP TABLE consumption",
        "INSERT INTO consumption VALUES ('2030-01-01', 1)",
        "UPDATE consumption SET consumption_mwh = 0",
        "CREATE TABLE t (x INT)",
    ],
)
def test_connection_is_read_only(raw_agent_connection, sql: str) -> None:
    with pytest.raises(duckdb.Error, match="read-only|read only"):
        raw_agent_connection.execute(sql)


def test_connection_cannot_read_files(raw_agent_connection, secret_file: Path) -> None:
    for sql in [
        f"SELECT * FROM read_csv('{secret_file.as_posix()}')",
        f"SELECT * FROM read_text('{secret_file.as_posix()}')",
        f"SELECT * FROM glob('{secret_file.parent.as_posix()}/*')",
    ]:
        with pytest.raises(duckdb.PermissionException):
            raw_agent_connection.execute(sql)


def test_connection_cannot_write_files_or_attach(raw_agent_connection, tmp_path: Path) -> None:
    for sql in [
        f"COPY (SELECT 1) TO '{(tmp_path / 'x.csv').as_posix()}'",
        f"ATTACH '{(tmp_path / 'x.duckdb').as_posix()}' AS x",
    ]:
        with pytest.raises(duckdb.PermissionException):
            raw_agent_connection.execute(sql)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "sql",
    [
        "SET enable_external_access = true",
        "SET memory_limit = '100GB'",
        "SET autoinstall_known_extensions = true",
        "RESET enable_external_access",
    ],
)
def test_connection_settings_are_locked(raw_agent_connection, sql: str) -> None:
    with pytest.raises(duckdb.Error, match="configuration"):
        raw_agent_connection.execute(sql)


def test_executor_requires_existing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SafeSqlExecutor(tmp_path / "missing.duckdb")
