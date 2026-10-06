"""HTTP API with fake agents: no LLM, no network."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from gridquery.agent.schemas import AgentResult, AgentStatus, Attempt, Usage
from gridquery.api.app import create_app
from gridquery.config import Settings
from gridquery.tools.charts import ChartSpec

SQL = "SELECT month_local, avg_price_eur_per_mwh FROM monthly_summary"
DATA_RANGE = (date(2023, 9, 25), date(2026, 9, 30))


def answered(question: str) -> AgentResult:
    return AgentResult(
        question=question,
        status=AgentStatus.ANSWERED,
        answer=f"Prices averaged 64.7 EUR/MWh.\n\n```sql\n{SQL}\n```",
        sql=SQL,
        columns=["month_local", "avg_price_eur_per_mwh"],
        rows=[["2024-03-01", 64.7], ["2024-04-01", 62.1]],
        row_count=2,
        chart=ChartSpec(type="line", x="month_local", y=["avg_price_eur_per_mwh"]),
        assumptions=["period: 2024-03"],
        attempts=[
            Attempt(number=1, sql="SELECT bad", ok=False, error="unknown column"),
            Attempt(number=2, sql=SQL, ok=True, row_count=2),
        ],
        timings_ms={"write_sql": 900.0, "total": 2500.0},
        usage=Usage(llm_calls=4, input_tokens=4000, output_tokens=300, cost_usd=0.0008),
    )


class FakeAgent:
    instances: list["FakeAgent"] = []

    def __init__(self) -> None:
        self.questions: list[str] = []
        self.closed = False
        FakeAgent.instances.append(self)

    def ask(self, question: str) -> AgentResult:
        self.questions.append(question)
        return answered(question)

    def data_range(self) -> tuple[date, date]:
        return DATA_RANGE

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_model="test-model", api_max_concurrency=2)


@pytest.fixture
def client(settings):
    FakeAgent.instances = []
    with TestClient(create_app(settings, FakeAgent, with_ui=False)) as test_client:
        yield test_client


def test_health_reports_model_and_data_period(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "provider": "openai",
        "model": "test-model",
        "data_from": "2023-09-25",
        "data_to": "2026-09-30",
        "detail": None,
    }


def test_ask_returns_answer_sql_results_chart_and_timing(client) -> None:
    response = client.post("/ask", json={"question": "Average price in March 2024?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Prices averaged 64.7 EUR/MWh."  # SQL is its own field
    assert body["sql"] == SQL
    assert body["results"] == {
        "columns": ["month_local", "avg_price_eur_per_mwh"],
        "rows": [["2024-03-01", 64.7], ["2024-04-01", 62.1]],
        "row_count": 2,
        "truncated": False,
    }
    assert body["chart"]["type"] == "line" and body["chart"]["x"] == "month_local"
    assert [a["ok"] for a in body["attempts"]] == [False, True]
    assert body["timings_ms"]["total"] == 2500.0
    assert body["usage"]["cost_usd"] == 0.0008
    assert "error" not in body


@pytest.mark.parametrize("question", ["", "x" * 1_001])
def test_invalid_questions_are_rejected_before_the_agent(client, question: str) -> None:
    assert client.post("/ask", json={"question": question}).status_code == 422
    assert all(agent.questions == [] for agent in FakeAgent.instances)


def test_pool_has_one_agent_per_concurrent_request_and_closes_them(settings) -> None:
    FakeAgent.instances = []
    with TestClient(create_app(settings, FakeAgent, with_ui=False)) as client:
        client.post("/ask", json={"question": "q"})
        assert len(FakeAgent.instances) == 2
    assert all(agent.closed for agent in FakeAgent.instances)


def test_startup_failure_is_reported_not_crashed(settings) -> None:
    def broken() -> FakeAgent:
        raise FileNotFoundError("Database data/gridquery.duckdb not found.")

    with TestClient(create_app(settings, broken, with_ui=False)) as client:
        health = client.get("/health")
        ask = client.post("/ask", json={"question": "q"})

    assert health.status_code == 503
    assert health.json()["status"] == "unavailable"
    assert "Database data/gridquery.duckdb not found" in health.json()["detail"]
    assert ask.status_code == 503
    assert "Database data/gridquery.duckdb not found" in ask.json()["detail"]


def test_root_redirects_to_docs_without_ui(client) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.headers["location"] == "/docs"


def test_ui_is_mounted(settings, monkeypatch) -> None:
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    with TestClient(create_app(settings, FakeAgent)) as client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/ui"
        page = client.get("/ui/")
    assert page.status_code == 200
    assert "<html" in page.text.lower()
