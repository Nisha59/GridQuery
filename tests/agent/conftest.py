"""Fixtures that build agents and nodes around the scripted FakeLLM."""

from collections.abc import Callable

import pytest
from agent_helpers import TODAY, FakeLLM

from gridquery.agent.graph import GridQueryAgent
from gridquery.agent.nodes import AgentNodes
from gridquery.tools.schema_context import SchemaContextProvider
from gridquery.tools.sql_executor import SafeSqlExecutor


@pytest.fixture
def make_agent(executor: SafeSqlExecutor) -> Callable[..., tuple[GridQueryAgent, FakeLLM]]:
    def make(*outputs, max_attempts: int = 3, **kwargs) -> tuple[GridQueryAgent, FakeLLM]:
        llm = FakeLLM(*outputs, **kwargs)
        agent = GridQueryAgent(
            llm, executor, max_attempts=max_attempts, explain_max_rows=5, today=lambda: TODAY
        )
        return agent, llm

    return make


@pytest.fixture
def make_nodes(executor: SafeSqlExecutor) -> Callable[..., tuple[AgentNodes, FakeLLM]]:
    def make(*outputs, max_attempts: int = 3) -> tuple[AgentNodes, FakeLLM]:
        llm = FakeLLM(*outputs)
        nodes = AgentNodes(
            llm,
            executor,
            SchemaContextProvider(executor),
            max_attempts=max_attempts,
            explain_max_rows=5,
            today=lambda: TODAY,
        )
        return nodes, llm

    return make
