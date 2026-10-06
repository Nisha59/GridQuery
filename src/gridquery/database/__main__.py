"""Command-line entry point: `uv run python -m gridquery.database {load,check}`.

load   Rebuild the database from data/raw/, then run the quality checks.
check  Run the quality checks on the existing database (read-only).
"""

import argparse
import logging
import sys

from gridquery.config import get_settings
from gridquery.database.connection import connect
from gridquery.database.loader import build_database
from gridquery.database.quality import has_errors, run_checks
from gridquery.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the requested command and return a process exit code."""
    parser = argparse.ArgumentParser(description="Build and check the GridQuery database.")
    parser.add_argument("command", choices=["load", "check"])
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command == "load":
        build_database(settings.raw_data_dir, settings.duckdb_path)

    try:
        with connect(settings.duckdb_path, read_only=True) as con:
            results = run_checks(con)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    return 1 if has_errors(results) else 0


if __name__ == "__main__":
    sys.exit(main())
