"""The LangGraph state passed between nodes.

Nodes return partial updates. Lists of attempts, timings and usage are merged
by reducers, so each node only reports what it added.
"""

import operator
from datetime import date
from typing import Annotated, TypedDict

from gridquery.agent.schemas import (
    AgentStatus,
    Attempt,
    SqlDraft,
    Understanding,
    Usage,
)
from gridquery.tools.charts import ChartSpec
from gridquery.tools.sql_executor import QueryResult


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Add up durations per node (a node can run several times)."""
    merged = dict(left)
    for name, duration in right.items():
        merged[name] = merged.get(name, 0.0) + duration
    return merged


class AgentState(TypedDict, total=False):
    """State of one question moving through the graph."""

    # Input
    question: str
    today: date
    schema_context: str
    data_period: str
    data_range: tuple[date, date]
    # Step results
    understanding: Understanding
    draft: SqlDraft
    result: QueryResult
    issues: list[str]
    chart: ChartSpec | None
    answer: str
    status: AgentStatus
    # Accumulated records
    attempts: Annotated[list[Attempt], operator.add]
    timings_ms: Annotated[dict[str, float], merge_timings]
    usage: Annotated[Usage, operator.add]
