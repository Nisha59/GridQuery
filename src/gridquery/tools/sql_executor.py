"""Safe execution of agent SQL against DuckDB.

Every query passes through these layers, in order:

1. sqlglot validation (sql_validator): one read-only SELECT over known tables.
2. DuckDB's own parser: the regenerated SQL must be exactly one SELECT statement.
3. A hardened, read-only connection: no writes, no file/network access, no
   extension installs, settings locked, memory capped.
4. Resource limits: a timeout that interrupts long queries and a row cap.

Failures never raise; they come back as a QueryResult with an error message
the agent can read and use to fix its query.
"""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb

from gridquery.config import Settings
from gridquery.database.connection import connect
from gridquery.tools.sql_validator import SqlValidationError, validate_sql

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 1_000


class ErrorKind(StrEnum):
    """Why a query failed."""

    VALIDATION = "validation"  # Rejected before running.
    TIMEOUT = "timeout"  # Interrupted after the time limit.
    EXECUTION = "execution"  # DuckDB raised an error (e.g. unknown column).


@dataclass(frozen=True)
class QueryResult:
    """Structured outcome of one query.

    Attributes:
        sql: The SQL that was submitted.
        columns: Column names, in order.
        rows: Result rows (at most the row limit).
        row_count: Number of rows returned in `rows`.
        truncated: True if the query produced more rows than the limit.
        execution_time_ms: Wall-clock time spent validating and running the query.
        error: Human-readable error message, or None on success.
        error_kind: Category of the error, or None on success.
    """

    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool
    execution_time_ms: float
    error: str | None = None
    error_kind: ErrorKind | None = None

    @property
    def ok(self) -> bool:
        """True if the query ran successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "sql": self.sql,
            "ok": self.ok,
            "columns": self.columns,
            "rows": [[_json_value(value) for value in row] for row in self.rows],
            "row_count": self.row_count,
            "truncated": self.truncated,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error": self.error,
            "error_kind": self.error_kind.value if self.error_kind else None,
        }


def agent_connection_config(memory_limit: str) -> dict[str, str | bool]:
    """DuckDB settings for the agent's connection.

    `lock_configuration` comes last in spirit: once the connection is open, SQL can
    no longer change any of these settings.
    """
    return {
        "enable_external_access": False,  # No files, URLs, ATTACH, COPY or extension installs.
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "memory_limit": memory_limit,
        "lock_configuration": True,
    }


class SafeSqlExecutor:
    """Runs validated, read-only queries with a timeout and a row limit."""

    def __init__(
        self,
        db_path: Path,
        *,
        timeout_seconds: float = 10.0,
        max_rows: int = 10_000,
        memory_limit: str = "1GB",
    ) -> None:
        """Open a hardened read-only connection to the database.

        Args:
            db_path: Path of the GridQuery DuckDB file.
            timeout_seconds: Queries running longer than this are interrupted.
            max_rows: Maximum number of rows returned per query.
            memory_limit: DuckDB memory limit, e.g. "1GB".
        """
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self._con = connect(db_path, read_only=True, config=agent_connection_config(memory_limit))
        self.allowed_tables = self._list_relations()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create an executor using the database path and limits from settings."""
        return cls(
            settings.duckdb_path,
            timeout_seconds=settings.query_timeout_seconds,
            max_rows=settings.max_result_rows,
            memory_limit=settings.query_memory_limit,
        )

    def __enter__(self) -> Self:
        """Return the executor for use in a with-block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def run(self, sql: str) -> QueryResult:
        """Validate and run one query; never raises for bad SQL.

        Args:
            sql: SQL proposed by the agent.

        Returns:
            Rows and metadata on success, or an error message and kind on failure.
        """
        started = time.perf_counter()
        try:
            validated = validate_sql(sql, self.allowed_tables)
            self._check_with_duckdb_parser(validated.sql)
        except SqlValidationError as exc:
            logger.warning("Rejected query: %s | %s", exc, _one_line(sql))
            return _failure(sql, started, str(exc), ErrorKind.VALIDATION)

        try:
            columns, rows, truncated = self._execute(validated.sql)
        except duckdb.InterruptException:
            message = (
                f"Query timed out after {self.timeout_seconds:g} s. "
                "Aggregate more (e.g. use daily_summary or monthly_summary) "
                "or filter the date range."
            )
            logger.warning("Timed out: %s", _one_line(sql))
            return _failure(sql, started, message, ErrorKind.TIMEOUT)
        except duckdb.Error as exc:
            logger.info("Query failed: %s | %s", _first_line(exc), _one_line(sql))
            return _failure(sql, started, str(exc)[:MAX_ERROR_LENGTH], ErrorKind.EXECUTION)

        result = QueryResult(
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            execution_time_ms=_elapsed_ms(started),
        )
        logger.info(
            "Query ok: %d rows%s in %.1f ms",
            result.row_count,
            " (truncated)" if truncated else "",
            result.execution_time_ms,
        )
        return result

    @contextmanager
    def internal_cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield a cursor for fixed queries written by GridQuery itself (e.g. schema lookup).

        It uses the same hardened read-only connection, but skips validation, so it must
        never be given SQL that came from the LLM or a user.
        """
        cursor = self._con.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _check_with_duckdb_parser(self, sql: str) -> None:
        """Second opinion from DuckDB's own parser: exactly one SELECT statement.

        Guards against sqlglot and DuckDB reading the same text differently.
        """
        try:
            statements = self._con.extract_statements(sql)
        except duckdb.Error as exc:
            raise SqlValidationError(f"SQL syntax error: {_first_line(exc)}") from exc
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise SqlValidationError("Only a single read-only SELECT statement is allowed.")

    def _execute(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]], bool]:
        """Run SQL on a fresh cursor, interrupting it after the timeout.

        A new cursor per query means no state (such as temp objects) carries over.
        """
        cursor = self._con.cursor()
        timer = threading.Timer(self.timeout_seconds, cursor.interrupt)
        timer.start()
        try:
            cursor.execute(sql)
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchmany(self.max_rows + 1)
        finally:
            timer.cancel()
            cursor.close()
        truncated = len(rows) > self.max_rows
        return columns, rows[: self.max_rows], truncated

    def _list_relations(self) -> frozenset[str]:
        """Return the names of all user tables and views (a trusted internal query)."""
        rows = self._con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE NOT internal "
            "UNION SELECT view_name FROM duckdb_views() WHERE NOT internal"
        ).fetchall()
        return frozenset(name for (name,) in rows)


def _failure(sql: str, started: float, message: str, kind: ErrorKind) -> QueryResult:
    """Build a QueryResult for a failed query."""
    return QueryResult(
        sql=sql,
        columns=[],
        rows=[],
        row_count=0,
        truncated=False,
        execution_time_ms=_elapsed_ms(started),
        error=message,
        error_kind=kind,
    )


def _json_value(value: Any) -> Any:
    """Convert DuckDB result values to JSON-friendly types."""
    if isinstance(value, datetime | date | dt_time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _elapsed_ms(started: float) -> float:
    """Milliseconds since `started` (a perf_counter value)."""
    return (time.perf_counter() - started) * 1000


def _first_line(exc: Exception) -> str:
    """Return the first line of an exception message."""
    return str(exc).strip().splitlines()[0]


def _one_line(sql: str, limit: int = 200) -> str:
    """Collapse SQL onto one line for logs."""
    return " ".join(sql.split())[:limit]
