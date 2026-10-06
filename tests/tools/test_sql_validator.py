import pytest

from gridquery.tools.sql_validator import SqlValidationError, validate_sql

TABLES = {"consumption", "generation", "day_ahead_prices", "energy_sources", "daily_summary"}


def assert_blocked(sql: str, match: str | None = None) -> None:
    with pytest.raises(SqlValidationError, match=match):
        validate_sql(sql, TABLES)


# --- Allowed queries -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM consumption",
        "select count(*) from consumption;",
        "SELECT 1",
        "  -- comment\nSELECT max(consumption_mwh) FROM consumption /* trailing */",
        "FROM consumption SELECT max(consumption_mwh)",
        "WITH x AS (SELECT * FROM consumption) SELECT count(*) FROM x",
        "WITH a AS (SELECT 1 AS v), b AS (SELECT v + 1 AS v FROM a) SELECT * FROM b",
        "SELECT timestamp_utc FROM consumption "
        "UNION ALL SELECT timestamp_utc FROM day_ahead_prices",
        "SELECT * FROM main.consumption",
        "SELECT s.name_de, sum(g.generation_mwh) FROM generation g "
        "JOIN energy_sources s USING (source) GROUP BY ALL",
        "SELECT to_berlin_time(timestamp_utc), avg(price_eur_per_mwh) OVER (ORDER BY timestamp_utc "
        "ROWS 23 PRECEDING) FROM day_ahead_prices",
        "SELECT * FROM daily_summary WHERE date_local BETWEEN '2025-01-01' AND '2025-01-31'",
        "SELECT * FROM generate_series(1, 3)",
        "SELECT * FROM range(10)",
        "SELECT (SELECT max(consumption_mwh) FROM consumption) AS peak",
        "SELECT 'DROP TABLE consumption' AS harmless_text",
        "SELECT count(*) FROM consumption WHERE consumption_mwh > (SELECT avg(consumption_mwh) "
        "FROM consumption)",
    ],
)
def test_read_only_selects_are_allowed(sql: str) -> None:
    validated = validate_sql(sql, TABLES)
    assert validated.sql  # regenerated SQL is what will run


def test_validated_query_lists_tables_but_not_ctes() -> None:
    validated = validate_sql(
        "WITH x AS (SELECT * FROM consumption) "
        "SELECT * FROM x JOIN generation USING (timestamp_utc)",
        TABLES,
    )
    assert validated.tables == {"consumption", "generation"}


def test_table_names_are_case_insensitive() -> None:
    assert validate_sql("SELECT * FROM Consumption", TABLES).tables == {"consumption"}


# --- Writes and schema changes ---------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE consumption",
        "DROP VIEW daily_summary",
        "DELETE FROM consumption",
        "DELETE FROM consumption WHERE 1=1",
        "INSERT INTO consumption VALUES ('2026-01-01', 1)",
        "INSERT INTO consumption SELECT * FROM consumption",
        "UPDATE consumption SET consumption_mwh = 0",
        "CREATE TABLE t AS SELECT * FROM consumption",
        "CREATE TEMP TABLE t AS SELECT 1",
        "CREATE VIEW v AS SELECT 1",
        "CREATE MACRO evil() AS 1",
        "ALTER TABLE consumption ADD COLUMN x INT",
        "ALTER TABLE consumption RENAME TO c2",
        "TRUNCATE consumption",
        "MERGE INTO consumption USING consumption AS s ON true WHEN MATCHED THEN DELETE",
        "SELECT * INTO new_table FROM consumption",
    ],
)
def test_writes_and_schema_changes_are_blocked(sql: str) -> None:
    assert_blocked(sql)


# --- Files, URLs, other databases, extensions --------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('secret.csv')",
        "SELECT * FROM read_csv_auto('secret.csv')",
        "SELECT * FROM read_parquet('data/*.parquet')",
        "SELECT * FROM read_json_auto('x.json')",
        "SELECT * FROM read_text('C:/Windows/win.ini')",
        "SELECT * FROM read_blob('.env')",
        "SELECT content FROM read_text('.env')",
        "SELECT * FROM 'secret.csv'",
        "SELECT * FROM '.env'",
        "SELECT * FROM 'https://example.com/data.parquet'",
        "SELECT * FROM glob('*')",
        "SELECT * FROM sqlite_scan('other.db', 'users')",
        "SELECT * FROM postgres_scan('host=x', 'public', 'users')",
        "SELECT * FROM consumption JOIN read_csv('x.csv') USING (timestamp_utc)",
        "SELECT * FROM consumption WHERE consumption_mwh IN (SELECT * FROM read_csv('x.csv'))",
        "WITH x AS (SELECT * FROM read_csv('x.csv')) SELECT * FROM x",
        "COPY consumption TO 'dump.csv'",
        "COPY (SELECT * FROM consumption) TO 'dump.parquet' (FORMAT parquet)",
        "COPY consumption FROM 'evil.csv'",
        "EXPORT DATABASE 'dump_dir'",
        "IMPORT DATABASE 'dump_dir'",
        "ATTACH 'other.duckdb' AS other",
        "ATTACH ':memory:' AS scratch",
        "DETACH gridquery",
        "SELECT * FROM other.main.users",
        "INSTALL httpfs",
        "LOAD httpfs",
        "FORCE INSTALL httpfs FROM community",
    ],
)
def test_file_network_and_database_access_is_blocked(sql: str) -> None:
    assert_blocked(sql)


# --- Session, settings, system and dynamic SQL --------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SET enable_external_access = true",
        "SET memory_limit = '100GB'",
        "RESET ALL",
        "PRAGMA database_list",
        "PRAGMA enable_profiling",
        "CALL pragma_version()",
        "SELECT * FROM pragma_table_info('consumption')",
        "SELECT * FROM duckdb_settings()",
        "SELECT * FROM duckdb_secrets()",
        "SELECT current_setting('home_directory')",
        "SELECT getenv('OPENAI_API_KEY')",
        "SELECT * FROM query('DROP TABLE consumption')",
        "SELECT * FROM query_table('consumption')",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM system.main.duckdb_tables",
        "USE other",
        "BEGIN TRANSACTION",
        "COMMIT",
        "ROLLBACK",
        "CHECKPOINT",
        "VACUUM",
        "DESCRIBE consumption",
        "SUMMARIZE consumption",
        "SHOW TABLES",
        "EXPLAIN ANALYZE SELECT 1",
        "PREPARE p AS SELECT 1",
        "EXECUTE p",
    ],
)
def test_settings_system_and_dynamic_sql_are_blocked(sql: str) -> None:
    assert_blocked(sql)


# --- Multiple statements and tricks ----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE consumption",
        "SELECT 1; SELECT 2",
        "SELECT * FROM consumption; DELETE FROM consumption;",
        "SELECT 1;\n-- innocent comment\nDROP TABLE consumption",
        "DROP TABLE consumption; SELECT 1",
    ],
)
def test_multiple_statements_are_blocked(sql: str) -> None:
    assert_blocked(sql, "Exactly one SQL statement")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM users",
        "SELECT * FROM consumption JOIN secrets USING (id)",
        "WITH x AS (SELECT 1) SELECT * FROM y",
    ],
)
def test_unknown_tables_are_blocked_with_helpful_message(sql: str) -> None:
    assert_blocked(sql, "Unknown table .* Available tables: consumption")


@pytest.mark.parametrize("sql", ["", "   ", ";", "-- only a comment"])
def test_empty_queries_are_blocked(sql: str) -> None:
    assert_blocked(sql)


def test_syntax_errors_are_reported() -> None:
    assert_blocked("SELEC * FRM consumption", "check the syntax")
    assert_blocked("SELECT (1", "syntax error")


def test_overlong_queries_are_blocked() -> None:
    assert_blocked("SELECT " + " + ".join(["1"] * 5_000), "too long")


def test_error_messages_name_the_problem() -> None:
    assert_blocked("DROP TABLE consumption", "DROP")
    assert_blocked("SELECT * FROM read_csv('x')", r"read_csv\(\) is not allowed")
    assert_blocked("SELECT getenv('HOME')", r"getenv\(\) is not allowed")
