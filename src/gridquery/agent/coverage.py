"""Check in code whether the data covers the periods a question asks about.

The understand step only resolves *which* periods a question means; comparing
them with the available data is done here, because language models are
unreliable at date arithmetic (they would declare a year inside the range
"not available").
"""

from dataclasses import dataclass
from datetime import date

from gridquery.agent.schemas import Language, Period


@dataclass(frozen=True)
class Coverage:
    """How well the data covers the requested periods.

    Attributes:
        missing: Periods without any data; the question cannot be answered.
        partial: Periods only partly covered, each with the covered part (start, end).
    """

    missing: list[Period]
    partial: list[tuple[Period, date, date]]

    @property
    def answerable(self) -> bool:
        """True if every requested period has at least some data."""
        return not self.missing


def check_coverage(periods: list[Period], first: date, last: date) -> Coverage:
    """Compare requested periods with the available data range (inclusive dates).

    Args:
        periods: Periods the question asks about (empty = whole range, always covered).
        first: First German calendar day with data.
        last: Last German calendar day with data.
    """
    missing: list[Period] = []
    partial: list[tuple[Period, date, date]] = []
    for period in periods:
        start, end = sorted((period.start, period.end))
        if end < first or start > last:
            missing.append(period)
        elif start < first or end > last:
            partial.append((period, max(start, first), min(end, last)))
    return Coverage(missing, partial)


RANGE_WORDS: dict[Language, str] = {"en": "to", "de": "bis"}


def describe(period: Period, language: Language = "en") -> str:
    """'2024-01-01 to 2024-12-31' ('bis' in German), or a single day."""
    if period.start == period.end:
        return period.start.isoformat()
    return f"{period.start.isoformat()} {RANGE_WORDS[language]} {period.end.isoformat()}"
