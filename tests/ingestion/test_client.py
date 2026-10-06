import httpx
import pytest

from gridquery.ingestion.client import SmardClient, SmardError
from gridquery.ingestion.series import SERIES_BY_KEY

SPEC = SERIES_BY_KEY["consumption"]


def make_client(handler, sleeps: list[float] | None = None) -> SmardClient:
    sleeps = sleeps if sleeps is not None else []
    return SmardClient(
        "https://smard.test/chart_data",
        max_retries=2,
        backoff_seconds=1.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
    )


def test_urls_follow_smard_scheme() -> None:
    client = make_client(lambda request: httpx.Response(200, json={}))
    assert client.index_url(SPEC, "hour") == "https://smard.test/chart_data/410/DE/index_hour.json"
    assert (
        client.chunk_url(SPEC, "hour", 123)
        == "https://smard.test/chart_data/410/DE/410_DE_hour_123.json"
    )


def test_fetch_index_returns_sorted_timestamps() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"timestamps": [3, 1, 2]}))
    assert client.fetch_index(SPEC) == [1, 2, 3]


def test_retries_transient_errors_with_backoff() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200, json={"series": []})])
    sleeps: list[float] = []
    client = make_client(lambda request: next(responses), sleeps)

    assert client.fetch_chunk(SPEC, 1) == {"series": []}
    assert sleeps == [1.0, 0.0]  # one backoff, then the (zero) politeness delay


def test_retries_network_errors_then_gives_up() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    sleeps: list[float] = []
    client = make_client(handler, sleeps)

    with pytest.raises(SmardError, match="after 3 attempts"):
        client.fetch_chunk(SPEC, 1)
    assert sleeps == [1.0, 2.0]


def test_not_found_returns_none_without_retry() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    assert make_client(handler).fetch_chunk(SPEC, 1) is None
    assert len(calls) == 1


def test_client_errors_are_not_retried() -> None:
    client = make_client(lambda request: httpx.Response(400))
    with pytest.raises(SmardError):
        client.fetch_chunk(SPEC, 1)
