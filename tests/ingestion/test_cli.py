from datetime import date

import pytest

from gridquery.ingestion.__main__ import resolve_range
from gridquery.ingestion.series import SERIES, get_series


def test_resolve_range_uses_berlin_midnights() -> None:
    start, end = resolve_range(date(2026, 8, 1), date(2026, 9, 1), years=3)
    assert start.isoformat() == "2026-08-01T00:00:00+02:00"
    assert end.isoformat() == "2026-09-01T00:00:00+02:00"


def test_resolve_range_defaults_start_to_years_before_end() -> None:
    start, _ = resolve_range(None, date(2024, 2, 29), years=3)
    assert start.date() == date(2021, 2, 28)


def test_resolve_range_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError):
        resolve_range(date(2026, 9, 1), date(2026, 8, 1), years=3)


def test_series_catalog_is_complete_and_unique() -> None:
    assert len(SERIES) == 13
    assert len({spec.filter_id for spec in SERIES}) == 13
    assert get_series(None) == list(SERIES)
    with pytest.raises(ValueError, match="Unknown series"):
        get_series(["coal"])
