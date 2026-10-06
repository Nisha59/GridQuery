"""Build the DuckDB database from the raw SMARD files in data/raw/.

The database is always rebuilt from scratch: raw files are the source of truth,
and a full rebuild of ~350k rows takes seconds. The new database is written to a
temporary file and swapped in at the end, so readers never see a half-loaded
database and a failed load leaves the previous one untouched.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from gridquery.database.connection import create_schema
from gridquery.database.energy_sources import ENERGY_SOURCES
from gridquery.ingestion.collector import parse_observations
from gridquery.ingestion.raw_store import RawStore
from gridquery.ingestion.series import SERIES, Category, SeriesSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesLoadStats:
    """What happened to one series while loading."""

    series: str
    rows_loaded: int
    null_values_skipped: int
    duplicates_dropped: int
    conflicting_duplicates: int


def read_series(store: RawStore, spec: SeriesSpec) -> pd.DataFrame:
    """Read all raw chunks of a series into a frame.

    Returns:
        Columns timestamp_utc (naive UTC), value (float, may be NaN) and fetched_at.
    """
    records = [
        (obs.timestamp.replace(tzinfo=None), obs.value, chunk.fetched_at)
        for chunk in store.iter_chunks(spec)
        for obs in parse_observations(chunk.response)
    ]
    frame = pd.DataFrame(records, columns=["timestamp_utc", "value", "fetched_at"])
    return frame.astype({"timestamp_utc": "datetime64[us]", "value": "float64"})


def clean_series(frame: pd.DataFrame, series: str) -> tuple[pd.DataFrame, SeriesLoadStats]:
    """Drop missing values and duplicate hours, keeping the most recently fetched value.

    Missing values are not stored: an hour without a value simply has no row, which
    the missing-hours quality check reports.
    """
    nulls = int(frame["value"].isna().sum())
    frame = frame.dropna(subset=["value"]).sort_values(["timestamp_utc", "fetched_at"])

    repeated = frame[frame.duplicated("timestamp_utc", keep=False)]
    conflicts = int((repeated.groupby("timestamp_utc")["value"].nunique() > 1).sum())
    is_older_copy = frame.duplicated("timestamp_utc", keep="last")
    if conflicts:
        logger.warning(
            "%s: %d hours have different values in different raw files; using the newest",
            series,
            conflicts,
        )

    cleaned = frame[~is_older_copy][["timestamp_utc", "value"]].reset_index(drop=True)
    stats = SeriesLoadStats(series, len(cleaned), nulls, int(is_older_copy.sum()), conflicts)
    return cleaned, stats


def insert_energy_sources(con: duckdb.DuckDBPyConnection) -> None:
    """Fill the energy_sources reference table."""
    con.executemany(
        "INSERT INTO energy_sources VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                s.source,
                s.name_en,
                s.name_de,
                s.is_renewable,
                s.is_fossil,
                s.smard_filter_id,
                s.description,
            )
            for s in ENERGY_SOURCES
        ],
    )


def insert_series(con: duckdb.DuckDBPyConnection, spec: SeriesSpec, frame: pd.DataFrame) -> None:
    """Insert one cleaned series into its fact table."""
    con.register("staging", frame)
    try:
        if spec.category is Category.GENERATION:
            con.execute(
                "INSERT INTO generation SELECT timestamp_utc, ?, value FROM staging", [spec.key]
            )
        elif spec.category is Category.CONSUMPTION:
            con.execute("INSERT INTO consumption SELECT timestamp_utc, value FROM staging")
        elif spec.category is Category.PRICE:
            con.execute("INSERT INTO day_ahead_prices SELECT timestamp_utc, value FROM staging")
    finally:
        con.unregister("staging")


def load_database(
    con: duckdb.DuckDBPyConnection,
    store: RawStore,
    specs: tuple[SeriesSpec, ...] = SERIES,
) -> list[SeriesLoadStats]:
    """Create the schema in an empty database and load all series into it."""
    create_schema(con)
    insert_energy_sources(con)
    all_stats = []
    for spec in specs:
        frame, stats = clean_series(read_series(store, spec), spec.key)
        if frame.empty:
            logger.warning("%s: no raw data found in %s", spec.key, store.root / spec.key)
        insert_series(con, spec, frame)
        logger.info(
            "%s: loaded %d rows (skipped %d missing values, %d duplicates)",
            spec.key,
            stats.rows_loaded,
            stats.null_values_skipped,
            stats.duplicates_dropped,
        )
        all_stats.append(stats)
    return all_stats


def build_database(raw_dir: Path, db_path: Path) -> list[SeriesLoadStats]:
    """Rebuild the database file at db_path from raw files, replacing it atomically."""
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building %s from %s", db_path, raw_dir)

    try:
        with duckdb.connect(str(tmp_path)) as con:
            stats = load_database(con, RawStore(raw_dir))
            con.execute("CHECKPOINT")
        os.replace(tmp_path, db_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Database ready: %s", db_path)
    return stats
