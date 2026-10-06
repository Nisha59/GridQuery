"""Fixtures for evaluation tests."""

from collections.abc import Callable

import pytest
from eval_helpers import FakeAgent

from gridquery.agent.schemas import AgentResult
from gridquery.evaluation.runner import Mode, ReferenceClock


@pytest.fixture
def fake_factory():
    """Factory building FakeAgents from one respond function; keeps the agents."""

    def make(respond: Callable[[str], AgentResult]):
        agents: dict[str, FakeAgent] = {}

        def factory(mode: Mode, clock: ReferenceClock) -> FakeAgent:
            agents[mode.name] = FakeAgent(respond, clock)
            return agents[mode.name]

        factory.agents = agents
        return factory

    return make
