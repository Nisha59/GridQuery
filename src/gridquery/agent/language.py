"""Cheap, deterministic English/German detection.

Used as a cross-check on the LLM: small models sometimes reply in the wrong
language. Only the two supported languages are distinguished, and only when
the evidence is clear; otherwise the result is None and the LLM's choice stands.
"""

import re

from gridquery.agent.schemas import Language

# fmt: off
GERMAN_WORDS = frozenset({
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem", "und", "oder",
    "ist", "sind", "war", "waren", "wurde", "wurden", "wird", "hat", "haben", "wie", "was",
    "wann", "wo", "welche", "welcher", "welches", "wieviel", "viel", "viele", "hoch",
    "höchste", "niedrigste", "im", "am", "vom", "zum", "zur", "für", "über", "unter", "mit",
    "von", "nach", "bei", "pro", "je", "auch", "nicht", "noch", "nur", "ich", "sie", "es",
    "man", "gibt", "zeig", "zeige", "zeigen", "durchschnitt", "durchschnittliche", "anteil",
    "strom", "preis", "preise", "erzeugung", "verbrauch", "monat", "monate", "jahr", "jahre",
    "tag", "tage", "stunde", "letzten", "letztes", "letzte", "heute", "gestern", "möchten",
    "können", "bitte", "zeitraum",
})
ENGLISH_WORDS = frozenset({
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "been", "has", "have", "had",
    "how", "what", "when", "where", "which", "much", "many", "high", "highest", "lowest", "in",
    "on", "at", "of", "to", "for", "from", "by", "with", "per", "also", "not", "only", "i",
    "you", "it", "there", "show", "average", "share", "electricity", "power", "price",
    "prices", "generation", "consumption", "month", "months", "year", "years", "day", "days",
    "hour", "last", "today", "yesterday", "did", "does", "do", "would", "could", "please",
    "period", "between",
})
# fmt: on
WORD = re.compile(r"[a-zäöüß]+")


def detect_language(text: str) -> Language | None:
    """Return "de" or "en" when the text clearly is one of them, otherwise None."""
    lowered = text.lower()
    words = WORD.findall(lowered)
    german = sum(word in GERMAN_WORDS for word in words) + 2 * len(re.findall(r"[äöüß]", lowered))
    english = sum(word in ENGLISH_WORDS for word in words)
    if german >= 2 and german > 2 * english:
        return "de"
    if english >= 2 and english > 2 * german:
        return "en"
    return None
