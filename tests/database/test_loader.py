from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from gridquery.database import loader
from gridquery.database.connection import connect
from gridquery.database.energy_sources import ENERGY_SOURCES
from gridquery.database.loader import build_database, clean_series, load_database
from gridquery.ingestion.raw_store import RawStore
from gridquery.ingestion.series import SERIES, Category

START = datetime(2026, 1, 4, 23, tzinfo=UTC)  # Monday 00:00 in Berlin
SPECS = tuple(s for s in SERIES if s.key in {"solar", "consumption", "day_ahead_price"})


def test_every_generation_series_has_energy_source_metadata() -> None:
    generation_keys = {s.key for s in SERIES if s.category is Category.GENERATION}
    assert {s.source for s in ENERGY_SOURCES} == generation_keys


def test_clean_series_drops_nulls_and_keeps_newest_duplicate() -> None:
    t0, t1, t2 = (datetime(2026, 1, 1, h) for h in range(3))
    old, new = datetime(2026, 1, 2), datetime(2026, 1, 3)
    frame = pd.DataFrame(
        {
            "timestamp_utc": [t0, t0, t1, t1, t2],
            "value": [1.0, 2.0, 5.0, 5.0, None],
            "fetched_at": [new, old, old, new, old],
        }
    )

    cleaned, stats = clean_series(frame, "solar")

    assert cleaned["value"].tolist() == [1.0, 5.0]  # t0 keeps the newer value
    assert (stats.rows_loaded, stats.null_values_skipped) == (2, 1)
    assert (stats.duplicates_dropped, stats.conflicting_duplicates) == (2, 1)


def test_load_database_fills_each_table(store: RawStore, save_chunk) -> None:
    save_chunk("solar", START, [0.0, 10.0, None])
    save_chunk("consumption", START, [40_000.0, 41_000.0])
    save_chunk("day_ahead_price", START, [-5.0, 80.0])

    with duckdb.connect() as con:
        stats = load_database(con, store, SPECS)

        assert {s.series: s.rows_loaded for s in stats} == {
            "solar": 2,
            "consumption": 2,
            "day_ahead_price": 2,
        }
        assert con.execute(
            "SELECT source, generation_mwh FROM generation ORDER BY 2"
        ).fetchall() == [
            ("solar", 0.0),
            ("solar", 10.0),
        ]
        assert con.execute("SELECT min(price_eur_per_mwh) FROM day_ahead_prices").fetchone() == (
            -5.0,
        )
        # Timestamps are stored as naive UTC.
        first = con.execute("SELECT min(timestamp_utc) FROM consumption").fetchone()[0]
        assert first == START.replace(tzinfo=None)


def test_load_merges_multiple_chunks(store: RawStore, save_chunk) -> None:
    save_chunk("consumption", START, [1.0] * 168)
    save_chunk("consumption", START + timedelta(weeks=1), [2.0] * 5)

    with duckdb.connect() as con:
        load_database(con, store, SPECS)
        totals = con.execute("SELECT count(*), sum(consumption_mwh) FROM consumption").fetchone()

    assert totals == (173, 178.0)


def test_build_database_creates_readable_file(tmp_path: Path, store: RawStore, save_chunk) -> None:
    save_chunk("consumption", START, [1.0, 2.0])
    db_path = tmp_path / "db" / "grid.duckdb"

    build_database(store.root, db_path)

    with connect(db_path) as con:
        assert con.execute("SELECT count(*) FROM consumption").fetchone() == (2,)
        assert con.execute("SELECT count(*) FROM energy_sources").fetchone() == (11,)
    assert not db_path.with_name("grid.duckdb.tmp").exists()


def test_failed_build_keeps_previous_database(
    tmp_path: Path, store: RawStore, save_chunk, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_chunk("consumption", START, [1.0])
    db_path = tmp_path / "grid.duckdb"
    build_database(store.root, db_path)

    def fail(*args: object) -> None:
        raise RuntimeError("load failed")

    monkeypatch.setattr(loader, "load_database", fail)
    with pytest.raises(RuntimeError):
        build_database(store.root, db_path)

    with connect(db_path) as con:
        assert con.execute("SELECT count(*) FROM consumption").fetchone() == (1,)
    assert not db_path.with_name("grid.duckdb.tmp").exists()


def test_read_only_connection_rejects_writes(tmp_path: Path, store: RawStore, save_chunk) -> None:
    save_chunk("consumption", START, [1.0])
    db_path = tmp_path / "grid.duckdb"
    build_database(store.root, db_path)

    with connect(db_path) as con, pytest.raises(duckdb.Error, match="read-only"):
        con.execute("DELETE FROM consumption")


def test_connect_read_only_missing_file_explains_how_to_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gridquery.database load"):
        connect(tmp_path / "missing.duckdb")
