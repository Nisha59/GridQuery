"""HTTP client for the SMARD chart_data JSON interface.

SMARD serves each series as fixed chunks (one week for hourly data). An index
file lists the start timestamp of every chunk:

    {base}/{filter}/{region}/index_{resolution}.json
    {base}/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json
"""

import logging
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

import httpx

from gridquery.ingestion.series import SeriesSpec

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class SmardError(Exception):
    """Raised when SMARD cannot deliver a response after all retries."""


class SmardClient:
    """Small SMARD API client with retries and exponential backoff."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        request_delay_seconds: float = 0.0,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client.

        Args:
            base_url: Base URL of the chart_data interface.
            timeout: Per-request timeout in seconds.
            max_retries: Retries after the first attempt for transient failures.
            backoff_seconds: Base delay; attempt n waits backoff_seconds * 2**n.
            request_delay_seconds: Pause after each successful request, to be polite.
            http_client: Injected httpx client (used by tests).
            sleep: Injected sleep function (used by tests).
        """
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._request_delay_seconds = request_delay_seconds
        self._http = http_client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._sleep = sleep

    def __enter__(self) -> Self:
        """Return the client for use in a with-block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client."""
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def index_url(self, spec: SeriesSpec, resolution: str) -> str:
        """Return the URL of a series' chunk index."""
        return f"{self._base_url}/{spec.filter_id}/{spec.region}/index_{resolution}.json"

    def chunk_url(self, spec: SeriesSpec, resolution: str, chunk_start_ms: int) -> str:
        """Return the URL of one data chunk."""
        name = f"{spec.filter_id}_{spec.region}_{resolution}_{chunk_start_ms}.json"
        return f"{self._base_url}/{spec.filter_id}/{spec.region}/{name}"

    def fetch_index(self, spec: SeriesSpec, resolution: str = "hour") -> list[int]:
        """Return the sorted chunk start timestamps (epoch ms) for a series.

        Raises:
            SmardError: If the index is missing or cannot be fetched.
        """
        payload = self.get_json(self.index_url(spec, resolution))
        if payload is None or "timestamps" not in payload:
            raise SmardError(f"No chunk index for {spec.key} ({spec.filter_id}/{spec.region})")
        return sorted(int(ts) for ts in payload["timestamps"])

    def fetch_chunk(
        self, spec: SeriesSpec, chunk_start_ms: int, resolution: str = "hour"
    ) -> dict[str, Any] | None:
        """Return the raw JSON of one chunk, or None if SMARD has no such chunk."""
        return self.get_json(self.chunk_url(spec, resolution, chunk_start_ms))

    def get_json(self, url: str) -> dict[str, Any] | None:
        """GET a JSON document, retrying transient failures.

        Returns:
            The decoded JSON object, or None on HTTP 404.

        Raises:
            SmardError: On non-retryable HTTP errors or when retries are exhausted.
        """
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.get(url)
                if response.status_code == 404:
                    return None
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    payload = response.json()
                    self._sleep(self._request_delay_seconds)
                    return payload
                reason = f"HTTP {response.status_code}"
            except httpx.HTTPStatusError as exc:
                raise SmardError(f"GET {url} failed: {exc}") from exc
            except (httpx.TransportError, ValueError) as exc:
                # Network problems, timeouts, and truncated/invalid JSON are transient.
                reason = f"{type(exc).__name__}: {exc}"

            if attempt < self._max_retries:
                delay = self._backoff_seconds * 2**attempt
                logger.warning(
                    "GET %s failed (%s); retry %d/%d in %.1fs",
                    url,
                    reason,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)

        raise SmardError(f"GET {url} failed after {self._max_retries + 1} attempts: {reason}")
