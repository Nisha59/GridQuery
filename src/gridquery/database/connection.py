"""DuckDB connection helpers."""

from importlib.resources import files
from pathlib import Path

import duckdb

from gridquery.database.descriptions import apply_comments, load_descriptions


def connect(
    path: Path,
    *,
    read_only: bool = True,
    config: dict[str, str | bool | int] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open the GridQuery database; read-only unless explicitly requested.

    Args:
        path: Path of the DuckDB file.
        read_only: Open without write access. The agent must always use True.
        config: Extra DuckDB configuration options.

    Raises:
        FileNotFoundError: If a read-only connection is requested for a missing file.
    """
    if read_only and not path.exists():
        raise FileNotFoundError(
            f"Database {path} not found. Build it with: uv run python -m gridquery.database load"
        )
    return duckdb.connect(str(path), read_only=read_only, config=config or {})


def schema_sql() -> str:
    """Return the DDL in schema.sql."""
    return files("gridquery.database").joinpath("schema.sql").read_text(encoding="utf-8")


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables, views, and macros in an empty database, with comments."""
    con.execute(schema_sql())
    apply_comments(con, load_descriptions())
