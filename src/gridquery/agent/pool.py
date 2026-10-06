"""A fixed pool of agents for serving several questions at the same time.

One agent is not meant to answer two questions concurrently: it shares one
DuckDB connection between its steps. The pool holds `size` independent agents
(each with its own read-only connection); a request borrows one and returns it.
Requests beyond `size` wait for a free agent.
"""

import logging
import queue
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from typing import Protocol

from gridquery.agent.schemas import AgentResult

logger = logging.getLogger(__name__)


class PooledAgent(Protocol):
    """What the pool needs from an agent (GridQueryAgent fits)."""

    def ask(self, question: str) -> AgentResult:
        """Answer one question."""
        ...

    def data_range(self) -> tuple[date, date]:
        """First and last day with data."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class AgentPool:
    """Hands out agents one request at a time."""

    def __init__(self, factory: Callable[[], PooledAgent], size: int) -> None:
        """Create `size` agents up front, so configuration errors show at startup.

        Raises:
            Exception: Whatever the factory raises (e.g. missing database or API key);
                agents created before the failure are closed.
        """
        self.size = size
        self._idle: queue.Queue[PooledAgent] = queue.Queue()
        self._all: list[PooledAgent] = []
        try:
            for _ in range(size):
                agent = factory()
                self._all.append(agent)
                self._idle.put(agent)
        except Exception:
            self.close()
            raise
        logger.info("Agent pool ready with %d agents", size)

    @contextmanager
    def acquire(self) -> Iterator[PooledAgent]:
        """Borrow an agent, waiting until one is free."""
        agent = self._idle.get()
        try:
            yield agent
        finally:
            self._idle.put(agent)

    def ask(self, question: str) -> AgentResult:
        """Answer a question with the next free agent."""
        with self.acquire() as agent:
            return agent.ask(question)

    def data_range(self) -> tuple[date, date]:
        """First and last day with data (same for every agent)."""
        with self.acquire() as agent:
            return agent.data_range()

    def close(self) -> None:
        """Close every agent created by the pool."""
        for agent in self._all:
            agent.close()
        self._all.clear()
