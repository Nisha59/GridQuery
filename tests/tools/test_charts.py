"""Chart specs: validation, fallback rules, and rendering of every chart type."""

from datetime import date
from decimal import Decimal

import plotly.graph_objects as go
import pytest

from gridquery.tools.charts import (
    SOURCE_NOTES,
    ChartSpec,
    axis_label,
    build_figure,
    chart_problems,
    column_kinds,
    fallback_chart,
    sanitize_text,
    validate_chart,
)

# Small results shaped like real agent queries.
MONTHLY = (
    ["month_local", "solar_twh", "wind_twh"],
    [["2026-01-01", 1.0, 10.0], ["2026-02-01", 2.0, 9.0], ["2026-03-01", 4.0, 8.0]],
)
MIX = (
    ["month_local", "source", "generation_twh"],
    [
        ["2026-01-01", "Solar", 1.0],
        ["2026-01-01", "Wind", 10.0],
        ["2026-02-01", "Solar", 2.0],
        ["2026-02-01", "Wind", 9.0],
    ],
)
SOURCES = (["source", "generation_twh"], [["Solar", 70.0], ["Lignite", 60.0], ["Wind", 100.0]])
DAILY = (
    ["renewable_share_pct", "avg_price_eur_per_mwh"],
    [[40.0, 120.0], [60.0, 90.0], [80.0, 40.0], [55.0, 100.0]],
)
HOURLY = (
    ["hour_of_day", "month", "avg_price_eur_per_mwh"],
    [[h, m, 50.0 + h - m] for m in (1, 2, 3) for h in range(24)],
)


def draw(spec: ChartSpec, data: tuple[list[str], list[list]]) -> go.Figure:
    return build_figure(spec, *data)


def source_note(figure: go.Figure) -> str:
    return figure.layout.annotations[-1].text


# --- Every chart type renders with title, labels with units and the data source ---------------


def test_line_chart() -> None:
    spec = ChartSpec(type="line", x="month_local", y=["solar_twh", "wind_twh"], title="Monthly")
    figure = draw(spec, MONTHLY)

    assert [t.type for t in figure.data] == ["scatter", "scatter"]  # Plotly lines are scatter
    assert figure.data[0].mode == "lines"
    assert [t.name for t in figure.data] == ["Solar (TWh)", "Wind (TWh)"]
    assert figure.layout.title.text == "Monthly"
    assert figure.layout.xaxis.title.text == "Month local"
    assert figure.layout.yaxis.title.text == "Value (TWh)"
    assert source_note(figure) == SOURCE_NOTES["en"]


def test_line_chart_x_axis_uses_real_dates() -> None:
    figure = draw(ChartSpec(type="line", x="month_local", y=["solar_twh"]), MONTHLY)
    assert str(figure.data[0].x[0]).startswith("2026-01-01")
    assert figure.layout.yaxis.title.text == "Solar (TWh)"


def test_area_chart_stacks_series() -> None:
    spec = ChartSpec(type="area", x="month_local", y=["generation_twh"], series="source")
    figure = draw(spec, MIX)

    assert {t.name for t in figure.data} == {"Solar", "Wind"}
    assert all(t.stackgroup for t in figure.data)
    assert figure.layout.yaxis.title.text == "Generation (TWh)"


def test_bar_chart() -> None:
    spec = ChartSpec(
        type="bar",
        x="source",
        y=["generation_twh"],
        title="Generation 2025",
        x_label="Energy source",
        y_label="Generation",
    )
    figure = draw(spec, SOURCES)

    assert figure.data[0].type == "bar"
    assert list(figure.data[0].x) == ["Solar", "Lignite", "Wind"]
    assert figure.layout.xaxis.title.text == "Energy source"
    assert figure.layout.yaxis.title.text == "Generation (TWh)"  # unit added by code


def test_grouped_bar_chart_with_several_y() -> None:
    figure = draw(ChartSpec(type="bar", x="month_local", y=["solar_twh", "wind_twh"]), MONTHLY)
    assert len(figure.data) == 2
    assert figure.layout.barmode == "group"


def test_stacked_bar_chart() -> None:
    spec = ChartSpec(type="stacked_bar", x="month_local", y=["generation_twh"], series="source")
    figure = draw(spec, MIX)

    assert [t.type for t in figure.data] == ["bar", "bar"]
    assert figure.layout.barmode == "relative"


def test_scatter_chart() -> None:
    spec = ChartSpec(type="scatter", x="renewable_share_pct", y=["avg_price_eur_per_mwh"])
    figure = draw(spec, DAILY)

    assert figure.data[0].mode == "markers"
    assert figure.layout.xaxis.title.text == "Renewable share (%)"
    assert figure.layout.yaxis.title.text == "Average price (EUR/MWh)"


def test_pie_chart() -> None:
    spec = ChartSpec(type="pie", x="source", y=["generation_twh"], title="Mix")
    figure = draw(spec, SOURCES)

    assert figure.data[0].type == "pie"
    assert list(figure.data[0].labels) == ["Solar", "Lignite", "Wind"]
    assert "TWh" in figure.data[0].hovertemplate
    assert source_note(figure) == SOURCE_NOTES["en"]


def test_heatmap_chart() -> None:
    spec = ChartSpec(type="heatmap", x="hour_of_day", y=["avg_price_eur_per_mwh"], series="month")
    figure = draw(spec, HOURLY)

    assert figure.data[0].type == "heatmap"
    assert len(figure.data[0].z) == 3 and len(figure.data[0].z[0]) == 24  # 3 months x 24 hours
    assert figure.layout.xaxis.title.text == "Hour of day"
    assert figure.layout.yaxis.title.text == "Month"
    assert figure.layout.coloraxis.colorbar.title.text == "Average price (EUR/MWh)"


def test_german_chart_has_german_source_and_default_title() -> None:
    spec = ChartSpec(type="bar", x="source", y=["generation_twh"], language="de")
    figure = draw(spec, SOURCES)
    assert source_note(figure) == SOURCE_NOTES["de"]
    assert figure.layout.title.text == "Generation (TWh) nach Source"


def test_missing_title_gets_a_default() -> None:
    figure = draw(ChartSpec(type="bar", x="source", y=["generation_twh"]), SOURCES)
    assert figure.layout.title.text == "Generation (TWh) by Source"


def test_decimal_values_render() -> None:
    rows = [["A", Decimal("1.5")], ["B", Decimal("2.5")]]
    figure = build_figure(ChartSpec(type="bar", x="k", y=["v_mwh"]), ["k", "v_mwh"], rows)
    assert list(figure.data[0].y) == [1.5, 2.5]


def test_build_figure_refuses_invalid_spec() -> None:
    with pytest.raises(ValueError, match="unknown column"):
        draw(ChartSpec(type="bar", x="nope", y=["generation_twh"]), SOURCES)


# --- Validation per type -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "data", "problem"),
    [
        (ChartSpec(type="line", x="missing", y=["solar_twh"]), MONTHLY, "unknown column"),
        (ChartSpec(type="line", x="month_local", y=["month_local"]), MONTHLY, "numeric"),
        (ChartSpec(type="line", x="source", y=["generation_twh"]), SOURCES, "time or numeric"),
        (ChartSpec(type="area", x="month_local", y=["solar_twh"]), MONTHLY, "area needs"),
        (ChartSpec(type="stacked_bar", x="source", y=["generation_twh"]), SOURCES, "stacked_bar"),
        (ChartSpec(type="scatter", x="month_local", y=["solar_twh"]), MONTHLY, "numeric x"),
        (ChartSpec(type="pie", x="month_local", y=["solar_twh", "wind_twh"]), MONTHLY, "pie"),
        (
            ChartSpec(type="heatmap", x="hour_of_day", y=["avg_price_eur_per_mwh"]),
            HOURLY,
            "heatmap",
        ),
        (
            ChartSpec(
                type="line", x="month_local", y=["generation_twh", "source"], series="source"
            ),
            MIX,
            "numeric",
        ),
    ],
)
def test_invalid_specs_are_explained(spec: ChartSpec, data, problem: str) -> None:
    problems = chart_problems(spec, *data)
    assert any(problem in p for p in problems), problems
    assert validate_chart(spec, *data) is None


def test_pie_rejects_negative_values_and_too_many_slices() -> None:
    negative = (["k", "v"], [["a", 1.0], ["b", -1.0]])
    assert "negative" in chart_problems(ChartSpec(type="pie", x="k", y=["v"]), *negative)[0]
    many = (["k", "v"], [[str(i), 1.0] for i in range(13)])
    assert "slices" in chart_problems(ChartSpec(type="pie", x="k", y=["v"]), *many)[0]


def test_too_many_bars_are_rejected() -> None:
    data = (["k", "v"], [[str(i), 1.0] for i in range(41)])
    assert "too many bars" in chart_problems(ChartSpec(type="bar", x="k", y=["v"]), *data)[0]


def test_single_row_is_never_charted() -> None:
    spec = ChartSpec(type="bar", x="source", y=["generation_twh"])
    assert validate_chart(spec, SOURCES[0], SOURCES[1][:1]) is None
    assert fallback_chart(SOURCES[0], SOURCES[1][:1]) is None


# --- Fallback rules ----------------------------------------------------------------------------


def test_fallback_time_series_is_a_line() -> None:
    chart = fallback_chart(*MONTHLY, title="Monthly", language="de")
    assert (chart.type, chart.x, chart.y, chart.language) == (
        "line",
        "month_local",
        ["solar_twh", "wind_twh"],
        "de",
    )


def test_fallback_time_series_with_category_uses_series() -> None:
    chart = fallback_chart(*MIX)
    assert (chart.type, chart.x, chart.y, chart.series) == (
        "line",
        "month_local",
        ["generation_twh"],
        "source",
    )


def test_fallback_categories_are_a_bar() -> None:
    assert fallback_chart(*SOURCES) == ChartSpec(type="bar", x="source", y=["generation_twh"])


def test_fallback_ordered_numbers_are_a_line_else_scatter() -> None:
    hours = (["hour_of_day", "avg_price_eur_per_mwh"], [[h, 50.0 + h] for h in range(24)])
    assert fallback_chart(*hours).type == "line"
    assert fallback_chart(*DAILY).type == "scatter"


def test_fallback_without_numbers_has_no_chart() -> None:
    assert fallback_chart(["a", "b"], [["x", "y"], ["z", "w"]]) is None


# --- Labels, units and text safety -------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "label", "expected"),
    [
        ("avg_price_eur_per_mwh", None, "Average price (EUR/MWh)"),
        ("total_generation_twh", None, "Total generation (TWh)"),
        ("consumption_gwh", None, "Consumption (GWh)"),
        ("renewable_share_pct", None, "Renewable share (%)"),
        ("hour_of_day", None, "Hour of day"),
        ("consumption_gwh", "Stromverbrauch", "Stromverbrauch (GWh)"),
        ("consumption_gwh", "Verbrauch in GWh", "Verbrauch in GWh"),  # unit not repeated
    ],
)
def test_axis_labels_always_carry_units(column: str, label: str | None, expected: str) -> None:
    assert axis_label(column, label) == expected


def test_llm_text_is_stripped_of_markup() -> None:
    evil = '<a href="javascript:alert(1)">Click</a> <b>Prices</b>' + "x" * 200
    cleaned = sanitize_text(evil)
    assert "<" not in cleaned and "href" not in cleaned
    assert cleaned.startswith("Click Prices")
    assert len(cleaned) == 120

    spec = ChartSpec(type="bar", x="source", y=["generation_twh"], title="<script>x</script>Mix")
    assert draw(spec, SOURCES).layout.title.text == "xMix"


def test_column_kinds() -> None:
    rows = [[date(2026, 1, 1), "solar", 1.5, None, True, Decimal("2")]]
    assert column_kinds(["d", "s", "v", "n", "b", "dec"], rows) == {
        "d": "time",
        "s": "category",
        "v": "number",
        "n": "empty",
        "b": "category",
        "dec": "number",
    }
    assert column_kinds(*MONTHLY)["month_local"] == "time"  # ISO strings from to_dict()
