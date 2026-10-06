"""Evaluation cases: loading and validating `eval_set.jsonl`.

Each line is one case. Answerable cases carry a gold SQL query and its expected
result; ambiguous and unanswerable cases carry the expected behaviour instead.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ExpectedBehavior = Literal["answer", "clarify", "refuse"]


class ExpectedResult(BaseModel):
    """Gold result of an answerable case."""

    columns: list[str]
    rows: list[list[Any]]


class EvalCase(BaseModel):
    """One benchmark question.

    Attributes:
        id: Stable identifier, e.g. "lookup_01".
        category: Question type, e.g. "simple_lookup" or "ambiguous".
        language: "en" or "de".
        question: The question as a user would ask it.
        expected_behavior: "answer", "clarify" or "refuse".
        reference_date: The "today" the question is asked on (for "last year" etc.).
        gold_sql: Reference query (answerable cases only).
        expected_result: Result of the gold query (answerable cases only).
        required_columns: Gold columns that must be matched; default all of them.
            Use it to mark informational gold columns as optional.
        notes: Free-text notes for reviewers.
        reason: Why the case is ambiguous or unanswerable.
        acceptable_responses: Descriptions of good responses (for human review).
    """

    id: str
    category: str
    language: Literal["en", "de"]
    question: str
    expected_behavior: ExpectedBehavior
    reference_date: date
    gold_sql: str | None = None
    expected_result: ExpectedResult | None = None
    required_columns: list[str] | None = None
    notes: str = ""
    reason: str = ""
    acceptable_responses: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_gold(self) -> "EvalCase":
        """Answerable cases need a gold result; required columns must exist in it."""
        if self.expected_behavior == "answer":
            if self.expected_result is None or not self.expected_result.rows:
                raise ValueError(f"{self.id}: answerable case needs a non-empty expected_result")
            unknown = set(self.required_columns or []) - set(self.expected_result.columns)
            if unknown:
                raise ValueError(f"{self.id}: required_columns not in gold result: {unknown}")
        return self

    def fingerprint(self) -> str:
        """Short hash of what the agent sees (question, reference date).

        Saved answers stay valid when gold answers or notes are corrected; they are
        re-judged when the report is built. Editing the question invalidates them.
        """
        payload = json.dumps([self.question, self.reference_date.isoformat()], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_cases(path: Path) -> list[EvalCase]:
    """Read and validate all cases from a JSONL file.

    Raises:
        ValueError: If a line is invalid or an id is duplicated.
    """
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [case.id for case in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate case ids in {path}: {sorted(duplicates)}")
    return cases
