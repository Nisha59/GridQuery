# GridQuery: API + UI on port 8000. Build and run with `docker compose up --build`.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    GRADIO_ANALYTICS_ENABLED=False \
    GRIDQUERY_API_HOST=0.0.0.0 \
    GRIDQUERY_API_PORT=8000

WORKDIR /app

# Dependencies first (cached until pyproject.toml/uv.lock change), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Run as an unprivileged user; data/ is mounted as a volume.
RUN useradd --create-home --uid 1000 gridquery && mkdir -p /app/data && chown gridquery /app/data
USER gridquery

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "gridquery.api"]
