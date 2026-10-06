"""Serve the API and UI: `uv run python -m gridquery.api` (http://127.0.0.1:8000)."""

import os
import sys

import uvicorn

from gridquery.config import get_settings
from gridquery.logging_config import configure_logging


def main() -> int:
    """Start uvicorn with host and port from the settings."""
    settings = get_settings()
    configure_logging(settings.log_level)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry from the UI
    uvicorn.run(
        "gridquery.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
