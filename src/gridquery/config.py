"""Application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for GridQuery. Values come from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GRIDQUERY_",
        extra="ignore",
    )

    # Database
    duckdb_path: Path = Path("data/gridquery.duckdb")

    # SMARD data collection
    smard_base_url: str = "https://www.smard.de/app/chart_data"
    raw_data_dir: Path = Path("data/raw")
    collection_years: int = Field(default=3, ge=1)
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=4, ge=0)
    http_backoff_seconds: float = Field(default=1.0, ge=0)
    http_request_delay_seconds: float = Field(default=0.1, ge=0)

    # LLM
    llm_provider: Literal["openai", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Prices in USD per million tokens, used for cost reporting. Unset = cost unknown.
    llm_input_cost_per_mtok: float | None = Field(default=None, ge=0)
    llm_output_cost_per_mtok: float | None = Field(default=None, ge=0)

    # Agent behaviour and safety limits
    max_sql_attempts: int = Field(default=3, ge=1)
    explain_max_rows: int = Field(default=50, ge=1)
    max_question_length: int = Field(default=1_000, ge=10)
    query_timeout_seconds: float = Field(default=10.0, gt=0)
    max_result_rows: int = Field(default=10_000, gt=0)
    query_memory_limit: str = "1GB"

    # HTTP API and UI (python -m gridquery.api)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # Questions answered at the same time; each slot holds its own agent and DB connection.
    api_max_concurrency: int = Field(default=4, ge=1)

    # Evaluation
    eval_dataset_path: Path = Path("data/eval/eval_set.jsonl")
    eval_output_dir: Path = Path("data/eval/results")
    # Hard limit in USD for one evaluation run (summed over resumed sessions).
    eval_max_cost_usd: float = Field(default=2.0, ge=0)

    # Optional LangSmith tracing (standard LangSmith variable names, no prefix)
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="gridquery", validation_alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str | None = Field(default=None, validation_alias="LANGSMITH_ENDPOINT")

    # Logging
    log_level: str = "INFO"

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "langsmith_api_key",
        "langsmith_endpoint",
        "llm_input_cost_per_mtok",
        "llm_output_cost_per_mtok",
        mode="before",
    )
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        """Treat empty values from `.env` (e.g. `OPENAI_BASE_URL=`) as not set."""
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
