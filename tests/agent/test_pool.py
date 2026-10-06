"""The agent pool used by the API and UI."""

import threading
from datetime import date

import pytest

from gridquery.agent.pool import AgentPool
from gridquery.agent.schemas import AgentResult, AgentStatus


class Agent:
    def __init__(self) -> None:
        self.closed = False

    def ask(self, question: str) -> AgentResult:
        return AgentResult(question=question, status=AgentStatus.ANSWERED, answer=str(id(self)))

    def data_range(self) -> tuple[date, date]:
        return date(2024, 1, 1), date(2024, 12, 31)

    def close(self) -> None:
        self.closed = True


def test_pool_creates_agents_up_front_and_closes_them() -> None:
    created: list[Agent] = []

    def factory() -> Agent:
        created.append(Agent())
        return created[-1]

    pool = AgentPool(factory, 3)
    assert len(created) == 3
    assert pool.ask("q").question == "q"
    assert pool.data_range() == (date(2024, 1, 1), date(2024, 12, 31))
    pool.close()
    assert all(agent.closed for agent in created)


def test_an_agent_is_never_shared_by_two_requests() -> None:
    pool = AgentPool(Agent, 1)
    with pool.acquire():
        waiting = threading.Thread(target=pool.ask, args=("q",))
        waiting.start()
        waiting.join(timeout=0.2)
        assert waiting.is_alive()  # blocked until the agent is returned
    waiting.join(timeout=2)
    assert not waiting.is_alive()


def test_factory_failure_closes_agents_already_created() -> None:
    created: list[Agent] = []

    def factory() -> Agent:
        if len(created) == 2:
            raise FileNotFoundError("no database")
        created.append(Agent())
        return created[-1]

    with pytest.raises(FileNotFoundError):
        AgentPool(factory, 3)
    assert all(agent.closed for agent in created)
