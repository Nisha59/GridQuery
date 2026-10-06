import pytest

from gridquery.agent.language import detect_language
from gridquery.agent.nodes import _fmt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How much solar power was generated in 2025?", "en"),
        ("What was the share of each energy source in total generation?", "en"),
        ("Wie hoch war der Strompreis im August 2026?", "de"),
        ("Welche Energiequelle hat am meisten Strom erzeugt?", "de"),
        ("Für welchen Zeitraum möchten Sie das wissen?", "de"),
        ("Strompreis?", None),  # too little evidence
        ("SMARD 2025", None),
    ],
)
def test_detect_language(text: str, expected: str | None) -> None:
    assert detect_language(text) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (13_212_345.6, "13,212,345.6"),
        (2_400_000.0, "2,400,000"),
        (46.633708, "46.63"),
        (0.123456, "0.1235"),
        (-19.0, "-19"),
        (1234, "1,234"),
        (None, "NULL"),
        ("solar", "solar"),
    ],
)
def test_explain_numbers_are_written_in_full(value: object, expected: str) -> None:
    assert _fmt(value) == expected
