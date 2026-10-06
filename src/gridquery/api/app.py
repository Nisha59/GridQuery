"""FastAPI application: POST /ask, GET /health, and the Gradio UI under /ui.

Run with `uv run python -m gridquery.api` (or uvicorn with `--factory
gridquery.api.app:create_app`). Agents are created at startup in a pool; if that
fails (no database, no API key) the app still starts, /health reports why and
/ask answers 503, so the problem is visible instead of a crash loop.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse

from gridquery.agent.graph import GridQueryAgent
from gridquery.agent.pool import AgentPool, PooledAgent
from gridquery.agent.schemas import AgentResult
from gridquery.api.schemas import AskRequest, AskResponse, HealthResponse
from gridquery.config import Settings, get_settings

logger = logging.getLogger(__name__)

UI_PATH = "/ui"


class ServiceUnavailableError(RuntimeError):
    """The agents are not running; the message says why."""


def create_app(
    settings: Settings | None = None,
    agent_factory: Callable[[], PooledAgent] | None = None,
    *,
    with_ui: bool = True,
) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration (default: from the environment / `.env`).
        agent_factory: Creates one agent; default: the configured GridQueryAgent.
        with_ui: Mount the Gradio UI under /ui.
    """
    settings = settings or get_settings()
    factory = agent_factory or (lambda: GridQueryAgent.from_settings(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            pool = AgentPool(factory, settings.api_max_concurrency)
            app.state.data_range = pool.data_range()
            app.state.pool = pool
        except Exception as exc:  # Report any setup problem via /health instead of crashing.
            logger.exception("Agents could not be started")
            app.state.startup_error = f"{type(exc).__name__}: {exc}"
        yield
        if app.state.pool is not None:
            app.state.pool.close()

    app = FastAPI(
        title="GridQuery",
        description="Questions about German electricity data (SMARD), in English or German.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.pool = None
    app.state.data_range = None
    app.state.startup_error = "starting up"

    def ask_question(question: str) -> AgentResult:
        """Answer with a pooled agent (shared by the API and the UI)."""
        pool: AgentPool | None = app.state.pool
        if pool is None:
            raise ServiceUnavailableError(app.state.startup_error)
        return pool.ask(question)

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest) -> AskResponse:
        """Answer a question: text, SQL, result rows, chart specification and timings."""
        if len(body.question) > settings.max_question_length:
            raise HTTPException(
                422, f"Question too long (max {settings.max_question_length} characters)."
            )
        try:
            return AskResponse.from_result(ask_question(body.question))
        except ServiceUnavailableError as exc:
            raise HTTPException(503, f"Service unavailable: {exc}") from exc

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response) -> HealthResponse:
        """Whether the agents are ready (503 if not), the model and the data period."""
        ready = app.state.pool is not None
        if not ready:
            response.status_code = 503
        data_range = app.state.data_range
        return HealthResponse(
            status="ok" if ready else "unavailable",
            provider=settings.llm_provider,
            model=settings.llm_model,
            data_from=data_range[0] if data_range else None,
            data_to=data_range[1] if data_range else None,
            detail=None if ready else app.state.startup_error,
        )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(UI_PATH if with_ui else "/docs")

    if with_ui:
        import gradio as gr

        from gridquery.ui.app import build_ui

        blocks = build_ui(ask_question, concurrency=settings.api_max_concurrency)
        app = gr.mount_gradio_app(app, blocks, path=UI_PATH, footer_links=[])
    return app
