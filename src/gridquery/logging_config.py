"""Central logging setup for command-line entry points."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for a command-line run.

    Args:
        level: Log level name such as "INFO" or "DEBUG".
    """
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT)
    # httpx logs every request at INFO; keep our own messages readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
