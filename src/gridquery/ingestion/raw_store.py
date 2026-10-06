"""File-based store for raw SMARD responses under data/raw/.

Each chunk is saved exactly as SMARD returned it, wrapped with provenance
metadata (URL and fetch time). Files are written atomically so an interrupted
run never leaves a half-written file behind.
"""

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gridquery.ingestion.series import SeriesSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawChunk:
    """A stored SMARD chunk with its provenance."""

    path: Path
    source_url: str
    fetched_at: datetime
    response: dict[str, Any]


class RawStore:
    """Read and write raw chunk files, one folder per series."""

    def __init__(self, root: Path) -> None:
        """Create a store rooted at the given directory."""
        self.root = root

    def path_for(self, spec: SeriesSpec, resolution: str, chunk_start_ms: int) -> Path:
        """Return the file path for one chunk."""
        name = f"{spec.filter_id}_{spec.region}_{resolution}_{chunk_start_ms}.json"
        return self.root / spec.key / name

    def iter_chunks(self, spec: SeriesSpec, resolution: str = "hour") -> Iterator[RawChunk]:
        """Yield all readable stored chunks of a series, oldest first."""
        pattern = f"{spec.filter_id}_{spec.region}_{resolution}_*.json"
        paths = sorted(
            (self.root / spec.key).glob(pattern), key=lambda p: int(p.stem.split("_")[-1])
        )
        for path in paths:
            chunk = self._read(path)
            if chunk is not None:
                yield chunk

    def load(self, spec: SeriesSpec, resolution: str, chunk_start_ms: int) -> RawChunk | None:
        """Load a stored chunk, or return None if it is absent or unreadable."""
        return self._read(self.path_for(spec, resolution, chunk_start_ms))

    def _read(self, path: Path) -> RawChunk | None:
        """Read one chunk file, or return None if it is absent or unreadable."""
        if not path.exists():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            return RawChunk(
                path=path,
                source_url=document["source_url"],
                fetched_at=datetime.fromisoformat(document["fetched_at"]),
                response=document["response"],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Ignoring unreadable raw file %s (%s); will refetch", path, exc)
            return None

    def save(
        self,
        spec: SeriesSpec,
        resolution: str,
        chunk_start_ms: int,
        *,
        source_url: str,
        fetched_at: datetime,
        response: dict[str, Any],
    ) -> RawChunk:
        """Atomically write a chunk and return it."""
        path = self.path_for(spec, resolution, chunk_start_ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "series": spec.key,
            "filter_id": spec.filter_id,
            "region": spec.region,
            "resolution": resolution,
            "chunk_start_ms": chunk_start_ms,
            "source_url": source_url,
            "fetched_at": fetched_at.isoformat(),
            "response": response,
        }
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(document), encoding="utf-8")
        os.replace(tmp_path, path)
        return RawChunk(path, source_url, fetched_at, response)
