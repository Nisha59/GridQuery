from decimal import Decimal

import pytest

from gridquery.agent.checks import check_result, normalize_sql
from gridquery.agent.schemas import Attempt
from gridquery.tools.sql_executor import ErrorKind, QueryResult


def result(columns=("x",), rows=((1,),), sql="SELECT 1", truncated=False, **kwargs) -> QueryResult:
    return QueryResult(
        sql=sql,
        columns=list(columns),
        rows=[tuple(r) for r in rows],
        row_count=len(rows),
        truncated=truncated,
        execution_time_ms=1.0,
        **kwargs,
    )


def test_plausible_result_has_no_issues() -> None:
    rows = [(0.4, 55.0, -120.0, 1000.0)]
    columns = ["renewable_share", "renewable_share_pct", "avg_price_eur_per_mwh", "solar_mwh"]
    assert check_result(result(columns, rows)) == []


def test_errors_are_reported() -> None:
    failed = result(rows=(), error="Binder Error: no column", error_kind=ErrorKind.EXECUTION)
    assert check_result(failed) == ["execution error: Binder Error: no column"]


def test_empty_result_asks_for_a_check() -> None:
    [issue] = check_result(result(rows=()))
    assert "no rows" in issue and "same SQL" in issue


def test_empty_result_is_accepted_when_the_same_query_is_resubmitted() -> None:
    previous = Attempt(number=1, sql="select 1\n", ok=True, row_count=0)
    assert check_result(result(rows=(), sql="SELECT  1;"), [previous]) == []


def test_different_query_with_empty_result_is_still_flagged() -> None:
    previous = Attempt(number=1, sql="SELECT 2", ok=True, row_count=0)
    assert check_result(result(rows=()), [previous])


def test_all_null_values_are_flagged() -> None:
    assert "All values are NULL" in check_result(result(("a", "b"), [(None, None)]))[0]


@pytest.mark.parametrize(
    ("column", "value", "fragment"),
    [
        ("renewable_share", 1.5, "outside 0-1"),
        ("share_pct", 120.0, "outside 0-100"),
        ("renewable_share", -0.1, "outside 0-1"),
        ("avg_price_eur_per_mwh", 5_000.0, "market limits"),
        ("price_eur_per_mwh", -600.0, "market limits"),
        ("solar_mwh", -1.0, "negative energy"),
        ("total_generation_twh", -0.5, "negative energy"),
        ("renewable_generation_share", 2.0, "outside 0-1"),
    ],
)
def test_implausible_values_are_flagged(column: str, value: float, fragment: str) -> None:
    [issue] = check_result(result([column], [(value,)]))
    assert fragment in issue


def test_words_are_matched_whole_not_as_substrings() -> None:
    # "generation" contains "ratio", "pricey" contains "price": neither may trigger a rule.
    assert check_result(result(["generation_mwh", "pricey_index"], [(500.0, 9_999.0)])) == []


def test_decimal_values_are_checked() -> None:
    [issue] = check_result(result(["generation_mwh"], [(Decimal("-5.0"),)]))
    assert "negative energy" in issue


def test_non_numeric_and_boolean_columns_are_ignored() -> None:
    assert check_result(result(["share_label", "price_flag"], [("high", True)])) == []


def test_truncated_result_is_flagged() -> None:
    [issue] = check_result(result(truncated=True))
    assert "cut off" in issue


def test_normalize_sql() -> None:
    assert normalize_sql("  SELECT\n  1 ; ") == normalize_sql("select 1")
