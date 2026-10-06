"""Optional LangSmith tracing.

LangChain and LangGraph trace automatically when the LANGSMITH_* variables are
present in the process environment. pydantic-settings reads `.env` but does not
export it, so this module copies the tracing settings into os.environ.
"""

import logging
import os

from gridquery.config import Settings

logger = logging.getLogger(__name__)


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing if configured; otherwise make sure it is off.

    Returns:
        True if tracing is enabled.
    """
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    if settings.langsmith_api_key is None:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing disabled")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    logger.info("LangSmith tracing enabled (project %s)", settings.langsmith_project)
    return True
