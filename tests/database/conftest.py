from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from gridquery.database.connection import create_schema
from gridquery.database.loader import insert_energy_sources
from gridquery.ingestion.collector import to_epoch_ms
from gridquery.ingestion.raw_store import RawStore
from gridquery.ingestion.series import SERIES_BY_KEY

SaveChunk = Callable[..., None]


@pytest.fixture
def store(tmp_path: Path) -> RawStore:
    return RawStore(tmp_path / "raw")


@pytest.fixture
def save_chunk(store: RawStore) -> SaveChunk:
    """Write a raw chunk: save_chunk(key, start, values, fetched_at=...)."""

    def save(
        key: str,
        start: datetime,
        values: list[float | None],
        fetched_at: datetime = datetime(2026, 2, 1, tzinfo=UTC),
    ) -> None:
        series = [[to_epoch_ms(start + i * timedelta(hours=1)), v] for i, v in enumerate(values)]
        store.save(
            SERIES_BY_KEY[key],
            "hour",
            to_epoch_ms(start),
            source_url="https://smard.test",
            fetched_at=fetched_at,
            response={"meta_data": {}, "series": series},
        )

    return save


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """In-memory database with the schema and energy sources, but no data."""
    connection = duckdb.connect()
    create_schema(connection)
    insert_energy_sources(connection)
    yield connection
    connection.close()
