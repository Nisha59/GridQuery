from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

from gridquery.database.connection import create_schema
from gridquery.database.descriptions import (
    ColumnDescription,
    Descriptions,
    RelationDescription,
    apply_comments,
    load_descriptions,
)
from gridquery.tools import schema_context
from gridquery.tools.schema_context import (
    SchemaContextProvider,
    build_schema_context,
    find_undocumented,
    render_schema_context,
)
from gridquery.tools.sql_executor import SafeSqlExecutor


@pytest.fixture
def text(executor: SafeSqlExecutor) -> str:
    return SchemaContextProvider(executor).text()


def block(text: str, relation: str) -> str:
    """The rendered block of one relation."""
    start = text.index(f"\n{relation} (") + 1
    return text[start : text.index("\n\n", start)]


# --- The descriptions file stays in sync with the schema --------------------------------------


def test_descriptions_file_matches_schema() -> None:
    with duckdb.connect() as con:
        create_schema(con)
        assert find_undocumented(con, load_descriptions()) == []


def test_drift_is_detected() -> None:
    descriptions = load_descriptions()
    consumption = descriptions.relations["consumption"]
    broken = Descriptions(
        descriptions.description,
        descriptions.notes,
        {
            **{k: v for k, v in descriptions.relations.items() if k != "generation"},
            "consumption": RelationDescription(
                consumption.description,
                {**consumption.columns, "old_column": ColumnDescription("gone")},
            ),
        },
    )
    with duckdb.connect() as con:
        create_schema(con)
        assert find_undocumented(con, broken) == [
            "consumption.old_column: not in database",
            "generation: missing from descriptions",
        ]


def test_comments_are_stored_in_database_with_units() -> None:
    with duckdb.connect() as con:
        create_schema(con)
        comment = con.execute(
            "SELECT comment FROM duckdb_columns() "
            "WHERE table_name = 'consumption' AND column_name = 'consumption_mwh'"
        ).fetchone()[0]
    assert comment == "Electricity consumed in the hour. [MWh]"


def test_load_descriptions_from_custom_file(tmp_path: Path) -> None:
    path = tmp_path / "d.toml"
    path.write_text(
        '[database]\ndescription = "Test DB"\n'
        '[relations.t]\ndescription = "It\'s a table"\n'
        '[relations.t.columns]\nx = { unit = "MWh", description = "Value" }\n',
        encoding="utf-8",
    )
    descriptions = load_descriptions(path)

    assert descriptions.notes == []
    assert descriptions.relations["t"].columns["x"] == ColumnDescription("Value", "MWh")
    with duckdb.connect() as con:
        con.execute("CREATE TABLE t (x DOUBLE)")
        apply_comments(con, descriptions)  # the apostrophe must be escaped
        assert con.execute("SELECT comment FROM duckdb_tables()").fetchone() == ("It's a table",)


# --- What the agent sees ----------------------------------------------------------------------


def test_every_relation_is_listed_with_description(text: str, executor: SafeSqlExecutor) -> None:
    for name in executor.allowed_tables:
        assert f"\n{name} (" in text
    assert "Hourly total electricity consumption" in text


def test_columns_have_types_units_and_descriptions(text: str) -> None:
    consumption = block(text, "consumption")
    assert "  timestamp_utc TIMESTAMP: Start of the hour, UTC." in consumption
    assert "  consumption_mwh DOUBLE [MWh]: Electricity consumed in the hour." in consumption
    assert "price_eur_per_mwh DOUBLE [EUR/MWh]" in text


def test_tables_show_row_counts_and_utc_date_ranges(text: str) -> None:
    assert block(text, "consumption").startswith(
        "consumption (table; 48 rows; 2026-01-01 00:00 to 2026-01-02 23:00 UTC)"
    )


def test_views_show_local_date_ranges_without_row_counts(text: str) -> None:
    assert block(text, "daily_summary").startswith("daily_summary (view; 2026-01-01 to 2026-01-03)")


def test_relations_without_time_axis_have_no_range(text: str) -> None:
    assert block(text, "data_coverage").startswith("data_coverage (view):")


def test_example_rows_two_for_tables_one_for_views(text: str) -> None:
    assert block(text, "consumption").count(" ; ") == 1
    assert "  e.g. (2026-01-02 23:00, 50000)" in block(text, "consumption")
    assert block(text, "hourly_summary").count(" ; ") == 0


def test_example_rows_skip_rows_with_nulls(text: str) -> None:
    # The latest hourly_summary rows have no NULLs here, but the check must hold in general.
    examples = [line for line in text.splitlines() if line.startswith("  e.g.")]
    assert examples and not any("NULL" in line for line in examples)


def test_identical_columns_are_not_repeated(text: str) -> None:
    monthly = block(text, "monthly_summary")
    assert "  month_local DATE: First day of the month." in monthly
    assert "(other columns as in daily_summary)" in monthly
    assert "avg_price_eur_per_mwh" not in monthly
    assert "(other columns as in generation_daily)" in block(text, "generation_monthly")


def test_energy_sources_are_listed_in_english_and_german(text: str) -> None:
    assert "  lignite | Lignite | Braunkohle | fossil |" in text
    assert "  solar | Solar (photovoltaics) | Photovoltaik | renewable |" in text
    assert "  nuclear | Nuclear | Kernenergie | other |" in text


def test_sources_with_unusual_data_ranges_are_flagged(text: str) -> None:
    # The test database only has solar and lignite data.
    nuclear = next(line for line in text.splitlines() if line.startswith("  nuclear |"))
    assert nuclear.endswith("No data.")
    solar = next(line for line in text.splitlines() if line.startswith("  solar |"))
    assert "only until" not in solar


def test_sources_ending_early_show_their_last_hour() -> None:
    sources = [
        schema_context.SourceInfo(
            "nuclear",
            "Nuclear",
            "Kernenergie",
            False,
            False,
            "Shut down.",
            datetime(2023, 1, 1),
            datetime(2024, 1, 30, 10),
        ),
        schema_context.SourceInfo(
            "solar",
            "Solar",
            "Photovoltaik",
            True,
            False,
            "Panels.",
            datetime(2023, 1, 1),
            datetime(2026, 9, 30),
        ),
    ]
    lines = schema_context._render_sources(sources)
    assert lines[1].endswith("Shut down. Data only until 2024-01-30 10:00 UTC.")
    assert lines[2].endswith("Panels.")


def test_notes_are_included(text: str) -> None:
    assert "to_berlin_time(ts)" in text
    assert "negative prices are real" in text


def test_output_is_compact(text: str) -> None:
    # Rough token estimate (~4 characters per token); keep the schema prompt small.
    assert len(text) < 8_000


def test_edited_descriptions_change_the_output(executor: SafeSqlExecutor) -> None:
    descriptions = load_descriptions()
    descriptions.relations["consumption"] = RelationDescription(
        "EDITED TEXT", descriptions.relations["consumption"].columns
    )
    text = SchemaContextProvider(executor, descriptions).text()
    assert (
        "consumption (table; 48 rows; 2026-01-01 00:00 to 2026-01-02 23:00 UTC): EDITED TEXT"
        in text
    )


def test_context_is_built_once_and_cached(
    executor: SafeSqlExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real_build = schema_context.build_schema_context

    def counting_build(*args, **kwargs):
        calls.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(schema_context, "build_schema_context", counting_build)
    provider = SchemaContextProvider(executor)

    assert provider.text() == provider.text()
    assert len(calls) == 1


def test_undocumented_relations_still_appear(db_path: Path) -> None:
    descriptions = load_descriptions()
    del descriptions.relations["consumption"]
    with duckdb.connect(str(db_path), read_only=True) as con:
        context = build_schema_context(con, descriptions)
    text = render_schema_context(context)
    assert "\nconsumption (table; 48 rows" in text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "NULL"),
        (datetime(2026, 1, 2, 3, 4, 5), "2026-01-02 03:04"),
        (date(2026, 1, 2), "2026-01-02"),
        (True, "true"),
        (1234.5678, "1234.57"),
        (205.0, "205"),
        (0.123456, "0.1235"),
        (-19.0, "-19"),
        ("solar", "'solar'"),
        (42, "42"),
    ],
)
def test_value_formatting(value: object, expected: str) -> None:
    assert schema_context._fmt(value) == expected
