"""Comparing agent results with gold results."""

import pytest

from gridquery.evaluation.matching import compare_results, compare_values

GOLD_COLUMNS = ["source", "generation_twh"]
GOLD_ROWS = [["wind_onshore", 112.56], ["solar", 63.148]]


def match(columns, rows, **kwargs) -> bool:
    return compare_results(GOLD_COLUMNS, GOLD_ROWS, columns, rows, **kwargs).match


def test_identical_result_matches() -> None:
    result = compare_results(GOLD_COLUMNS, GOLD_ROWS, GOLD_COLUMNS, GOLD_ROWS)
    assert result.match
    assert result.column_map == {"source": "source", "generation_twh": "generation_twh"}


def test_column_names_column_order_row_order_and_extra_columns_are_ignored() -> None:
    rows = [[63.148, "solar", 1], [112.56, "wind_onshore", 2]]
    assert match(["twh", "src", "rank"], rows)


@pytest.mark.parametrize(
    "rows",
    [
        [["wind_onshore", 112_560_000.0], ["solar", 63_148_000.0]],  # MWh
        [["wind_onshore", 112_560.4], ["solar", 63_148.2]],  # GWh, unrounded
        [["wind_onshore", 112.5612], ["solar", 63.1477]],  # more decimals than gold
    ],
)
def test_units_and_extra_precision_are_accepted(rows) -> None:
    assert match(GOLD_COLUMNS, rows)


def test_percent_matches_fraction() -> None:
    assert compare_results(["share_pct"], [[60.2]], ["share"], [[0.60213]]).match


@pytest.mark.parametrize(("value", "expected"), [(64.74, True), (64.66, True), (64.76, False)])
def test_tolerance_follows_gold_precision(value: float, expected: bool) -> None:
    assert compare_results(["avg"], [[64.7]], ["avg"], [[value]]).match is expected


def test_counts_must_be_exact() -> None:
    assert not compare_results(["hours"], [[457]], ["n"], [[458]]).match


def test_wrong_value_is_reported() -> None:
    result = compare_results(
        GOLD_COLUMNS, GOLD_ROWS, GOLD_COLUMNS, [["wind_onshore", 112.56], ["solar", 70.0]]
    )
    assert not result.match
    assert "generation_twh" in result.reason


def test_rows_must_line_up_not_only_columns() -> None:
    swapped = [["wind_onshore", 63.148], ["solar", 112.56]]
    result = compare_results(GOLD_COLUMNS, GOLD_ROWS, GOLD_COLUMNS, swapped)
    assert not result.match
    assert "line up" in result.reason


def test_row_count_must_match_without_limit() -> None:
    result = compare_results(GOLD_COLUMNS, GOLD_ROWS, GOLD_COLUMNS, GOLD_ROWS[:1])
    assert not result.match
    assert "row count" in result.reason


def test_longer_ranking_matches_top_n_gold_query() -> None:
    rows = [*GOLD_ROWS, ["lignite", 70.9]]
    gold_sql = "SELECT ... ORDER BY generation_twh DESC LIMIT 2"
    assert match(GOLD_COLUMNS, rows, gold_sql=gold_sql)
    assert not match(GOLD_COLUMNS, rows)  # no LIMIT: the whole result must match
    assert not match(GOLD_COLUMNS, [["lignite", 70.9], *GOLD_ROWS], gold_sql=gold_sql)


def test_empty_agent_result_never_matches() -> None:
    assert not match(GOLD_COLUMNS, [])


@pytest.mark.parametrize(
    "agent_value",
    ["2025-01-20 17:00:00", "2025-01-20T17:00:00+01:00", "2025-01-20T17:00"],
)
def test_timestamp_formats(agent_value: str) -> None:
    assert compare_results(["ts"], [["2025-01-20T17:00:00"]], ["t"], [[agent_value]]).match


def test_date_matches_midnight_timestamp_but_not_other_dates() -> None:
    assert compare_results(["month"], [["2024-03-01"]], ["m"], [["2024-03-01T00:00:00"]]).match
    assert not compare_results(["month"], [["2024-03-01"]], ["m"], [["2024-04-01"]]).match


def test_strings_ignore_case_and_separators() -> None:
    assert match(GOLD_COLUMNS, [["Wind onshore", 112.56], ["SOLAR", 63.148]])


def test_numeric_strings_compare_as_numbers() -> None:
    assert compare_results(["year"], [[2024]], ["year"], [["2024"]]).match


def test_only_required_columns_are_checked() -> None:
    gold_columns = ["source", "generation_twh", "first_month"]
    gold_rows = [["wind_onshore", 43.528, "2023-09-01"]]
    agent_rows = [["wind_onshore", 43.528]]
    assert not compare_results(gold_columns, gold_rows, GOLD_COLUMNS, agent_rows).match
    assert compare_results(
        gold_columns,
        gold_rows,
        GOLD_COLUMNS,
        agent_rows,
        required_columns=["source", "generation_twh"],
    ).match


def test_zero_and_null_values() -> None:
    assert compare_results(["mwh"], [[0.0]], ["mwh"], [[0]]).match
    assert compare_results(["a", "b"], [[None, 1.0]], ["a", "b"], [[None, 1000.0]]).match
    assert not compare_results(["mwh"], [[0.0]], ["mwh"], [[None]]).match


# --- compare_values ------------------------------------------------------------------------


def values(gold_columns, gold_rows, agent_rows, **kwargs) -> bool:
    return compare_values(gold_columns, gold_rows, agent_rows, **kwargs).match


def test_values_ignore_labels_and_layout() -> None:
    gold_columns, gold_rows = ["year", "twh"], [[2025, 26.92], [2026, 33.292]]
    wide = [[26_920_227.75, 33_291_767.03]]  # one column per year, in MWh
    assert values(gold_columns, gold_rows, wide, required_columns=["twh"])
    month_labels = [["2025-01-01", 114.14], ["2026-01-01", 110.09]]
    assert values(["month", "avg"], month_labels, [[2025, 114.14], [2026, 110.09]])


def test_values_require_every_number_once() -> None:
    gold_rows = [["a", 10.0], ["b", 10.0]]
    assert not values(["k", "v"], gold_rows, [[10.0]])
    assert values(["k", "v"], gold_rows, [[10.0, 10.0]])
    assert not values(["k", "v"], gold_rows, [["a", 10.0], ["b", 11.0]])


def test_values_use_one_scale_for_the_whole_result() -> None:
    gold = [[1.5, 2.5]]
    assert values(["a", "b"], gold, [[1500.0, 2500.0]])
    result = compare_values(["a", "b"], gold, [[1500.0, 2.5]])
    assert not result.match
    assert "different unit scales" in result.reason


def test_values_require_timestamps_but_not_dates() -> None:
    gold_columns, gold_rows = ["ts", "price"], [["2025-01-20T17:00:00", 583.4]]
    assert values(gold_columns, gold_rows, [["2025-01-20 17:00:00", 583.4]])
    wrong_hour = compare_values(gold_columns, gold_rows, [["2025-01-20 18:00:00", 583.4]])
    assert not wrong_hour.match and "timestamp" in wrong_hour.reason


def test_values_reject_long_dumps() -> None:
    gold = [[26.195]]
    daily = [[0.07]] * 364 + [[26.195]]
    result = compare_values(["twh"], gold, daily)
    assert not result.match and "row count" in result.reason


def test_values_report_missing_number() -> None:
    result = compare_values(["twh"], [[26.195]], [[30.0]])
    assert not result.match and "26.195" in result.reason
