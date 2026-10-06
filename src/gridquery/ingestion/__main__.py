"""Command-line entry point: `uv run python -m gridquery.ingestion`.

Examples:
    uv run python -m gridquery.ingestion                    # all series, last N years
    uv run python -m gridquery.ingestion --series consumption --start 2026-08-01 --end 2026-09-01
    uv run python -m gridquery.ingestion --list-series
"""

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta

from gridquery.config import get_settings
from gridquery.ingestion.client import SmardClient
from gridquery.ingestion.collector import BERLIN, collect
from gridquery.ingestion.raw_store import RawStore
from gridquery.ingestion.series import SERIES, get_series
from gridquery.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download SMARD data into data/raw/.")
    parser.add_argument("--series", nargs="+", metavar="KEY", help="Series keys (default: all)")
    parser.add_argument("--start", type=date.fromisoformat, help="First day, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="Day after the last, YYYY-MM-DD")
    parser.add_argument("--years", type=int, help="Years back from --end if --start is omitted")
    parser.add_argument("--list-series", action="store_true", help="List series and exit")
    return parser.parse_args(argv)


def resolve_range(start: date | None, end: date | None, years: int) -> tuple[datetime, datetime]:
    """Turn optional CLI dates into a [start, end) range of German local midnights.

    Defaults: end is tomorrow (so today is included), start is `years` before end.
    """
    end_day = end or date.today() + timedelta(days=1)
    start_day = start or _years_before(end_day, years)
    if start_day >= end_day:
        raise ValueError(f"--start {start_day} must be before --end {end_day}")
    return _berlin_midnight(start_day), _berlin_midnight(end_day)


def _years_before(day: date, years: int) -> date:
    """Return the same calendar day `years` earlier (Feb 29 becomes Feb 28)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _berlin_midnight(day: date) -> datetime:
    """Return midnight at the start of `day` in German local time."""
    return datetime.combine(day, time.min, tzinfo=BERLIN)


def main(argv: list[str] | None = None) -> int:
    """Run the collection and return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args(argv)

    if args.list_series:
        for spec in SERIES:
            logger.info(
                "%-20s filter=%-5d region=%-6s %s",
                spec.key,
                spec.filter_id,
                spec.region,
                spec.label_de,
            )
        return 0

    try:
        specs = get_series(args.series)
        start, end = resolve_range(args.start, args.end, args.years or settings.collection_years)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    client = SmardClient(
        settings.smard_base_url,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        backoff_seconds=settings.http_backoff_seconds,
        request_delay_seconds=settings.http_request_delay_seconds,
    )
    with client:
        reports = collect(specs, start, end, client, RawStore(settings.raw_data_dir))

    failed = [report.series for report in reports if not report.ok]
    if failed:
        logger.error("Incomplete series (re-run to resume): %s", ", ".join(failed))
        return 1
    logger.info("Collection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
