from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from gridquery.database.connection import create_schema
from gridquery.database.loader import insert_energy_sources
from gridquery.tools.sql_executor import SafeSqlExecutor

T0 = datetime(2026, 1, 1)
HOURS = 48


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small GridQuery database file: two days of hourly data."""
    path = tmp_path_factory.mktemp("db") / "gridquery.duckdb"
    with duckdb.connect(str(path)) as con:
        create_schema(con)
        insert_energy_sources(con)
        hours = [T0 + timedelta(hours=h) for h in range(HOURS)]
        con.executemany(
            "INSERT INTO generation VALUES (?, ?, ?)",
            [(ts, source, 100.0) for ts in hours for source in ("solar", "lignite")],
        )
        con.executemany("INSERT INTO consumption VALUES (?, ?)", [(ts, 50_000.0) for ts in hours])
        con.executemany(
            "INSERT INTO day_ahead_prices VALUES (?, ?)",
            [(ts, float(h - 10)) for h, ts in enumerate(hours)],
        )
    return path


@pytest.fixture
def executor(db_path: Path) -> Iterator[SafeSqlExecutor]:
    with SafeSqlExecutor(db_path, timeout_seconds=5, max_rows=100) as ex:
        yield ex


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    """A file the agent must never be able to read."""
    path = tmp_path / "secret.csv"
    path.write_text("password\nhunter2\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never send traces to LangSmith."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
