"""Collect SMARD series into the raw store, resumably, and report what was collected.

Resumability: a chunk that was fetched well after its week ended is treated as
final and never downloaded again. Chunks that were still filling up when they
were fetched (the current week, or a week whose data was not yet published) are
fetched again on the next run.

Missing values: SMARD marks unpublished or unavailable hours as null. Raw files
keep those nulls unchanged; the report counts them so gaps are visible instead
of silently filled.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from gridquery.ingestion.client import SmardClient, SmardError
from gridquery.ingestion.raw_store import RawChunk, RawStore
from gridquery.ingestion.series import SeriesSpec

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
CHUNK_LENGTH = timedelta(weeks=1)
# SMARD fills in and corrects recent values for a while; treat a chunk as final only
# if it was fetched at least this long after its last hour.
SETTLE_PERIOD = timedelta(days=3)


@dataclass(frozen=True)
class Chunk:
    """Time span [start, end) covered by one SMARD chunk."""

    start: datetime
    end: datetime

    @property
    def start_ms(self) -> int:
        """Chunk start as epoch milliseconds (the SMARD chunk ID)."""
        return to_epoch_ms(self.start)


@dataclass(frozen=True)
class Observation:
    """One hourly value; value is None when SMARD has no data for that hour."""

    timestamp: datetime
    value: float | None


@dataclass
class SeriesReport:
    """Summary of collecting one series over a date range."""

    series: str
    chunks_total: int = 0
    chunks_fetched: int = 0
    chunks_cached: int = 0
    chunks_missing: int = 0
    chunks_failed: int = 0
    values: int = 0
    missing_values: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every chunk was either fetched or already stored."""
        return self.chunks_failed == 0 and not self.errors


def to_epoch_ms(moment: datetime) -> int:
    """Convert an aware datetime to epoch milliseconds."""
    return int(moment.timestamp() * 1000)


def from_epoch_ms(epoch_ms: int) -> datetime:
    """Convert epoch milliseconds to an aware UTC datetime."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)


def select_chunks(index: list[int], start: datetime, end: datetime) -> list[Chunk]:
    """Return the chunks from a SMARD index that overlap [start, end).

    A chunk ends where the next one starts; the last chunk is assumed to last a week.
    """
    starts = [from_epoch_ms(ts) for ts in sorted(index)]
    ends = starts[1:] + [starts[-1] + CHUNK_LENGTH] if starts else []
    return [
        Chunk(chunk_start, chunk_end)
        for chunk_start, chunk_end in zip(starts, ends, strict=True)
        if chunk_start < end and chunk_end > start
    ]


def is_final(chunk: Chunk, fetched_at: datetime) -> bool:
    """Return True if a chunk fetched at `fetched_at` will not change any more."""
    return fetched_at >= chunk.end + SETTLE_PERIOD


def parse_observations(response: dict[str, Any]) -> list[Observation]:
    """Parse a raw SMARD chunk into observations.

    Malformed points are logged and skipped; non-numeric values become None.
    """
    observations = []
    for point in response.get("series", []):
        try:
            timestamp_ms, raw_value = point
            timestamp = from_epoch_ms(int(timestamp_ms))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed SMARD point: %r", point)
            continue
        observations.append(Observation(timestamp, _to_float(raw_value)))
    return observations


def _to_float(raw_value: Any) -> float | None:
    """Convert a SMARD value to float, mapping null and junk to None."""
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning("Treating non-numeric SMARD value as missing: %r", raw_value)
        return None


def collect_series(
    spec: SeriesSpec,
    start: datetime,
    end: datetime,
    client: SmardClient,
    store: RawStore,
    *,
    resolution: str = "hour",
    now: datetime | None = None,
) -> SeriesReport:
    """Download all chunks of one series overlapping [start, end) into the raw store.

    Args:
        spec: Series to collect.
        start: Inclusive start of the range (timezone-aware).
        end: Exclusive end of the range (timezone-aware).
        client: SMARD API client.
        store: Raw file store.
        resolution: SMARD resolution.
        now: Current time; injectable for tests.

    Returns:
        A report with chunk counts, value counts, and the covered time range.
    """
    now = now or datetime.now(UTC)
    report = SeriesReport(series=spec.key)
    try:
        chunks = select_chunks(client.fetch_index(spec, resolution), start, end)
    except SmardError as exc:
        logger.error("%s: could not load chunk index: %s", spec.key, exc)
        report.errors.append(str(exc))
        return report

    report.chunks_total = len(chunks)
    if not chunks:
        logger.warning("%s: SMARD has no data between %s and %s", spec.key, start, end)

    observations: list[Observation] = []
    for chunk in chunks:
        raw = _load_or_fetch(spec, chunk, client, store, resolution, now, report)
        if raw is not None:
            observations.extend(parse_observations(raw.response))

    _summarize(report, observations, start, min(end, now))
    return report


def _load_or_fetch(
    spec: SeriesSpec,
    chunk: Chunk,
    client: SmardClient,
    store: RawStore,
    resolution: str,
    now: datetime,
    report: SeriesReport,
) -> RawChunk | None:
    """Return a chunk from the store if final, otherwise fetch and store it."""
    stored = store.load(spec, resolution, chunk.start_ms)
    if stored is not None and is_final(chunk, stored.fetched_at):
        report.chunks_cached += 1
        return stored

    try:
        response = client.fetch_chunk(spec, chunk.start_ms, resolution)
    except SmardError as exc:
        logger.error("%s: chunk %s failed: %s", spec.key, chunk.start.date(), exc)
        report.chunks_failed += 1
        return stored  # Fall back to an older, non-final copy if we have one.

    if response is None:
        logger.warning("%s: chunk %s listed in index but not found", spec.key, chunk.start)
        report.chunks_missing += 1
        return stored

    report.chunks_fetched += 1
    return store.save(
        spec,
        resolution,
        chunk.start_ms,
        source_url=client.chunk_url(spec, resolution, chunk.start_ms),
        fetched_at=now,
        response=response,
    )


def _summarize(
    report: SeriesReport,
    observations: list[Observation],
    start: datetime,
    past_end: datetime,
) -> None:
    """Fill value counts and covered range for observations in [start, past_end).

    Hours after `past_end` (i.e. in the future) are ignored, so values that simply
    have not happened yet are not counted as missing. Day-ahead prices for
    tomorrow are still kept in the raw files.
    """
    in_range = [obs for obs in observations if start <= obs.timestamp < past_end]
    present = [obs.timestamp for obs in in_range if obs.value is not None]
    report.values = len(present)
    report.missing_values = len(in_range) - len(present)
    if present:
        report.first_timestamp = min(present)
        report.last_timestamp = max(present)


def collect(
    specs: list[SeriesSpec],
    start: datetime,
    end: datetime,
    client: SmardClient,
    store: RawStore,
) -> list[SeriesReport]:
    """Collect several series and log a summary line for each."""
    logger.info(
        "Collecting %d series from %s to %s into %s",
        len(specs),
        start.isoformat(),
        end.isoformat(),
        store.root,
    )
    reports = []
    for spec in specs:
        report = collect_series(spec, start, end, client, store)
        log_report(report)
        reports.append(report)
    return reports


def log_report(report: SeriesReport) -> None:
    """Log a one-line summary of a series report."""
    covered = (
        f"{_fmt(report.first_timestamp)} to {_fmt(report.last_timestamp)}"
        if report.first_timestamp
        else "no data"
    )
    level = logging.INFO if report.ok else logging.ERROR
    logger.log(
        level,
        "%s: %d chunks (%d fetched, %d cached, %d missing, %d failed) | "
        "%d hourly values, %d missing | %s",
        report.series,
        report.chunks_total,
        report.chunks_fetched,
        report.chunks_cached,
        report.chunks_missing,
        report.chunks_failed,
        report.values,
        report.missing_values,
        covered,
    )


def _fmt(moment: datetime | None) -> str:
    """Format a timestamp in German local time for logs."""
    return moment.astimezone(BERLIN).strftime("%Y-%m-%d %H:%M %Z") if moment else "-"
