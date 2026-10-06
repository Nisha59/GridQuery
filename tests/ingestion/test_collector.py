from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from gridquery.ingestion.client import SmardClient
from gridquery.ingestion.collector import (
    Chunk,
    collect_series,
    is_final,
    parse_observations,
    select_chunks,
    to_epoch_ms,
)
from gridquery.ingestion.raw_store import RawStore
from gridquery.ingestion.series import SERIES_BY_KEY

SPEC = SERIES_BY_KEY["consumption"]
HOUR = timedelta(hours=1)
WEEK = timedelta(weeks=1)
W0 = datetime(2026, 1, 4, 23, tzinfo=UTC)  # Monday 00:00 Berlin
W1 = W0 + WEEK
W2 = W1 + WEEK


def week_series(start: datetime, value: float | None = 1.0) -> list[list]:
    return [[to_epoch_ms(start + i * HOUR), value] for i in range(168)]


class FakeSmard:
    """In-memory SMARD server backed by httpx.MockTransport."""

    def __init__(
        self, chunks: dict[datetime, list[list]], unavailable: tuple[datetime, ...] = ()
    ) -> None:
        self.chunks = {to_epoch_ms(start): series for start, series in chunks.items()}
        self.index = sorted([*self.chunks, *(to_epoch_ms(start) for start in unavailable)])
        self.chunk_requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name.startswith("index_"):
            return httpx.Response(200, json={"timestamps": self.index})
        self.chunk_requests += 1
        ts = int(name.removesuffix(".json").rsplit("_", 1)[-1])
        if ts not in self.chunks:
            return httpx.Response(404)
        return httpx.Response(200, json={"meta_data": {}, "series": self.chunks[ts]})

    def client(self) -> SmardClient:
        return SmardClient(
            "https://smard.test",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            sleep=lambda seconds: None,
        )


def test_select_chunks_keeps_only_overlapping_weeks() -> None:
    index = [to_epoch_ms(W0), to_epoch_ms(W1), to_epoch_ms(W2)]
    chunks = select_chunks(index, W1 + HOUR, W1 + 2 * HOUR)
    assert chunks == [Chunk(W1, W2)]


def test_select_chunks_assumes_last_chunk_is_one_week() -> None:
    chunks = select_chunks([to_epoch_ms(W0)], W0, W2)
    assert chunks == [Chunk(W0, W1)]


def test_is_final_requires_settle_period_after_chunk_end() -> None:
    chunk = Chunk(W0, W1)
    assert not is_final(chunk, W1)
    assert is_final(chunk, W1 + timedelta(days=3))


def test_parse_observations_keeps_nulls_and_skips_malformed_points() -> None:
    response = {"series": [[to_epoch_ms(W0), 5], [to_epoch_ms(W0 + HOUR), None], ["bad"]]}
    observations = parse_observations(response)
    assert [obs.value for obs in observations] == [5.0, None]
    assert observations[0].timestamp == W0


def test_collect_series_saves_raw_files_and_reports_counts(tmp_path: Path) -> None:
    series = week_series(W0)
    series[10][1] = None  # a gap in the middle
    fake = FakeSmard({W0: series, W1: week_series(W1)})
    store = RawStore(tmp_path)

    report = collect_series(SPEC, W0, W2, fake.client(), store, now=W2 + WEEK)

    assert report.ok
    assert (report.chunks_total, report.chunks_fetched, report.chunks_cached) == (2, 2, 0)
    assert report.values == 2 * 168 - 1
    assert report.missing_values == 1
    assert report.first_timestamp == W0
    assert report.last_timestamp == W2 - HOUR
    assert store.path_for(SPEC, "hour", to_epoch_ms(W0)).exists()


def test_collect_series_resumes_from_final_raw_files(tmp_path: Path) -> None:
    fake = FakeSmard({W0: week_series(W0), W1: week_series(W1)})
    store = RawStore(tmp_path)
    collect_series(SPEC, W0, W2, fake.client(), store, now=W2 + WEEK)
    fake.chunk_requests = 0

    report = collect_series(SPEC, W0, W2, fake.client(), store, now=W2 + 2 * WEEK)

    assert fake.chunk_requests == 0
    assert report.chunks_cached == 2
    assert report.values == 2 * 168


def test_collect_series_refetches_chunks_that_were_still_filling(tmp_path: Path) -> None:
    fake = FakeSmard({W0: week_series(W0, None)})
    store = RawStore(tmp_path)
    first = collect_series(SPEC, W0, W1, fake.client(), store, now=W0 + 2 * HOUR)
    assert first.values == 0

    fake.chunks[to_epoch_ms(W0)] = week_series(W0)
    second = collect_series(SPEC, W0, W1, fake.client(), store, now=W1 + WEEK)

    assert second.chunks_fetched == 1
    assert second.values == 168


def test_future_hours_are_not_counted_as_missing(tmp_path: Path) -> None:
    series = [[ts, value if i < 24 else None] for i, (ts, value) in enumerate(week_series(W0))]
    fake = FakeSmard({W0: series})

    report = collect_series(SPEC, W0, W1, fake.client(), RawStore(tmp_path), now=W0 + 24 * HOUR)

    assert (report.values, report.missing_values) == (24, 0)


def test_chunk_listed_but_missing_is_reported(tmp_path: Path) -> None:
    fake = FakeSmard({}, unavailable=(W0,))

    report = collect_series(SPEC, W0, W1, fake.client(), RawStore(tmp_path), now=W1 + WEEK)

    assert report.chunks_missing == 1
    assert report.values == 0


@pytest.mark.parametrize("corrupt", ["", "{not json", '{"source_url": "x"}'])
def test_unreadable_raw_files_are_refetched(tmp_path: Path, corrupt: str) -> None:
    fake = FakeSmard({W0: week_series(W0)})
    store = RawStore(tmp_path)
    path = store.path_for(SPEC, "hour", to_epoch_ms(W0))
    path.parent.mkdir(parents=True)
    path.write_text(corrupt, encoding="utf-8")

    report = collect_series(SPEC, W0, W1, fake.client(), store, now=W1 + WEEK)

    assert report.chunks_fetched == 1
    assert report.values == 168
