"""Chart specifications and their Plotly rendering.

Safety by design: the LLM never writes chart code. It returns a ChartSpec, plain
data naming a chart type and result columns. This module checks the spec against
the actual query result and draws it with one fixed function, build_figure().
Text from the LLM (titles, labels) is stripped of markup before it is rendered.

Every chart gets a title, axis labels with units, and the data source.
"""

import re
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field

ChartType = Literal["line", "area", "bar", "stacked_bar", "scatter", "pie", "heatmap"]
Language = Literal["en", "de"]

MAX_Y_COLUMNS = 4
MAX_BAR_CATEGORIES = 40
MAX_PIE_SLICES = 12
MAX_HEATMAP_SIDE = 100
MAX_TEXT_LENGTH = 120

# Shown to the LLM in the SQL prompt, so code and prompt describe the same chart types.
CHART_GUIDE = """\
- "line": trend over time or an ordered axis (x = time/hour column); several y columns or a \
series column for multiple lines.
- "area": composition over time (stacked), e.g. generation per source per month; needs a \
series column or several y columns.
- "bar": comparing categories or a few periods.
- "stacked_bar": parts of a total per category/period; needs a series column or several y.
- "scatter": relationship between two numeric columns (e.g. renewable share vs price).
- "pie": share of a whole with at most 12 parts and non-negative values; one y, no series.
- "heatmap": a value over two dimensions (x and series), e.g. average price by hour (x) and \
month (series); exactly one y.
- "none": a single value, or a chart adds nothing."""

SOURCE_NOTES: dict[Language, str] = {
    "en": "Source: Bundesnetzagentur | SMARD.de (CC BY 4.0)",
    "de": "Quelle: Bundesnetzagentur | SMARD.de (CC BY 4.0)",
}

# Column-name suffix -> unit, checked in order (longest first).
UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_eur_per_mwh", "EUR/MWh"),
    ("_eur_mwh", "EUR/MWh"),
    ("_twh", "TWh"),
    ("_gwh", "GWh"),
    ("_mwh", "MWh"),
    ("_gw", "GW"),
    ("_mw", "MW"),
    ("_percent", "%"),
    ("_pct", "%"),
)
WORD_NAMES = {"avg": "average", "min": "minimum", "max": "maximum", "utc": "(UTC)"}


class ChartSpec(BaseModel):
    """A chart described as data: which type, and which result columns go where."""

    type: ChartType
    x: str = Field(description="Column for the x axis (pie: slice names).")
    y: list[str] = Field(min_length=1, max_length=MAX_Y_COLUMNS, description="Value column(s).")
    series: str | None = Field(
        default=None,
        description="Category column that splits the data: one line/stack per value, or the "
        "rows of a heatmap.",
    )
    title: str = ""
    x_label: str | None = None
    y_label: str | None = None
    language: Language = "en"


# --- Validation -------------------------------------------------------------------------------


def column_kinds(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, str]:
    """Classify each column as 'time', 'number', 'category' or 'empty' from its values."""
    return {
        column: _kind([row[i] for row in rows if row[i] is not None])
        for i, column in enumerate(columns)
    }


def chart_problems(
    spec: ChartSpec, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> list[str]:
    """Explain why a spec cannot be drawn from this result; empty if it can."""
    if len(rows) < 2:
        return ["a chart needs at least two rows"]
    kinds = column_kinds(columns, rows)
    used = [spec.x, *spec.y, *([spec.series] if spec.series else [])]
    missing = [c for c in used if c not in kinds]
    if missing:
        return [f"unknown column(s): {', '.join(missing)}"]

    problems = []
    if any(kinds[y] != "number" for y in spec.y):
        problems.append("y columns must be numeric")
    if spec.x in spec.y or spec.series in spec.y:
        problems.append("x/series must not also be a y column")
    if spec.series and kinds[spec.series] not in ("category", "number", "time"):
        problems.append("series column has no values")
    if spec.series and len(spec.y) > 1 and spec.type != "heatmap":
        problems.append("use either a series column or several y columns, not both")
    problems += _type_problems(spec, kinds, columns, rows)
    return problems


def validate_chart(
    spec: ChartSpec, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> ChartSpec | None:
    """Return the spec if it can be drawn from this result, otherwise None."""
    return None if chart_problems(spec, columns, rows) else spec


def _type_problems(
    spec: ChartSpec, kinds: dict[str, str], columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> list[str]:
    """Rules specific to each chart type."""
    x_kind = kinds[spec.x]
    distinct_x = _distinct(columns, rows, spec.x)
    stacked = spec.series is not None or len(spec.y) > 1
    if spec.type in ("line", "area") and x_kind not in ("time", "number"):
        return [f"{spec.type} needs a time or numeric x axis"]
    if spec.type == "area" and not stacked:
        return ["area needs a series column or several y columns"]
    if spec.type in ("bar", "stacked_bar") and distinct_x > MAX_BAR_CATEGORIES:
        return [f"too many bars ({distinct_x} > {MAX_BAR_CATEGORIES})"]
    if spec.type == "stacked_bar" and not stacked:
        return ["stacked_bar needs a series column or several y columns"]
    if spec.type == "scatter" and x_kind != "number":
        return ["scatter needs a numeric x axis"]
    if spec.type == "pie":
        values = _numbers(columns, rows, spec.y[0])
        if len(spec.y) > 1 or spec.series:
            return ["pie takes exactly one y column and no series"]
        if distinct_x > MAX_PIE_SLICES:
            return [f"too many pie slices ({distinct_x} > {MAX_PIE_SLICES})"]
        if any(v < 0 for v in values):
            return ["pie values must not be negative"]
    if spec.type == "heatmap":
        if not spec.series or len(spec.y) != 1:
            return ["heatmap needs x, a series column (rows) and exactly one y"]
        if max(distinct_x, _distinct(columns, rows, spec.series)) > MAX_HEATMAP_SIDE:
            return ["heatmap is too large"]
    return []


# --- Fallback when the LLM's suggestion does not fit ------------------------------------------


def fallback_chart(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    title: str = "",
    language: Language = "en",
) -> ChartSpec | None:
    """Pick a simple chart by rule, or None if no chart makes sense."""
    if len(rows) < 2:
        return None
    kinds = column_kinds(columns, rows)
    numbers = [c for c in columns if kinds[c] == "number"]
    times = [c for c in columns if kinds[c] == "time"]
    categories = [c for c in columns if kinds[c] == "category"]
    common = {"title": title, "language": language}
    if times and numbers:
        series = categories[0] if categories else None
        y = numbers[:1] if series else numbers[:MAX_Y_COLUMNS]
        return ChartSpec(type="line", x=times[0], y=y, series=series, **common)
    if categories and numbers and _distinct(columns, rows, categories[0]) <= MAX_BAR_CATEGORIES:
        return ChartSpec(type="bar", x=categories[0], y=numbers[:MAX_Y_COLUMNS], **common)
    if len(numbers) >= 2:
        x = numbers[0]
        x_values = _numbers(columns, rows, x)
        ordered = x_values == sorted(set(x_values))  # e.g. hour_of_day 0..23
        return ChartSpec(type="line" if ordered else "scatter", x=x, y=numbers[1:2], **common)
    return None


# --- Rendering ---------------------------------------------------------------------------------


def build_figure(
    spec: ChartSpec, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> go.Figure:
    """Draw a validated spec with Plotly: the only place where charts are created.

    Raises:
        ValueError: If the spec does not fit the data.
    """
    problems = chart_problems(spec, columns, rows)
    if problems:
        raise ValueError(f"Cannot draw {spec.type} chart: {'; '.join(problems)}")

    kinds = column_kinds(columns, rows)
    frame = _frame(columns, rows, kinds)
    x_title = axis_label(spec.x, spec.x_label)
    y_title = _y_title(spec)
    title = sanitize_text(spec.title) or _default_title(spec, x_title, y_title)
    # Readable names for legends and hover text.
    names = {column: axis_label(column) for column in columns}
    frame = frame.rename(columns=names)
    x, series = names[spec.x], names.get(spec.series) if spec.series else None
    y = [names[c] for c in spec.y]
    y_arg: str | list[str] = y[0] if len(y) == 1 else y

    if spec.type == "pie":
        figure = px.pie(frame, names=x, values=y[0], title=title)
        figure.update_traces(
            sort=False, hovertemplate=f"%{{label}}: %{{value:,.2f}} {_unit(spec.y[0])}"
        )
    elif spec.type == "heatmap":
        grid = frame.pivot_table(index=series, columns=x, values=y[0], aggfunc="mean")
        figure = px.imshow(
            grid,
            aspect="auto",
            title=title,
            labels={"x": x_title, "y": axis_label(spec.series), "color": y_title},
        )
    else:
        plot = {
            "line": px.line,
            "area": px.area,
            "bar": px.bar,
            "stacked_bar": px.bar,
            "scatter": px.scatter,
        }[spec.type]
        kwargs: dict[str, Any] = {"x": x, "y": y_arg, "color": series, "title": title}
        if spec.type == "bar" and (series or len(y) > 1):
            kwargs["barmode"] = "group"
        if spec.type == "stacked_bar":
            kwargs["barmode"] = "relative"
        figure = plot(frame, **kwargs)
        figure.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    _finish(figure, spec.language)
    return figure


def axis_label(column: str, label: str | None = None) -> str:
    """Readable label for a column, always ending with its unit when one is known.

    Args:
        column: Result column name, e.g. "avg_price_eur_per_mwh".
        label: Optional label from the LLM (in the answer language).
    """
    unit = _unit(column)
    text = sanitize_text(label) if label else _humanize(column)
    if unit and unit.lower() not in text.lower():
        text = f"{text} ({unit})"
    return text


def sanitize_text(text: str | None) -> str:
    """Strip markup and control characters from LLM text before Plotly renders it."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", "", text)
    text = " ".join(text.replace("<", "").replace(">", "").split())
    return text[:MAX_TEXT_LENGTH]


def _y_title(spec: ChartSpec) -> str:
    """Y axis title: the LLM label or column name, with the (shared) unit."""
    if len(spec.y) == 1:
        return axis_label(spec.y[0], spec.y_label)
    units = {_unit(c) for c in spec.y}
    unit = units.pop() if len(units) == 1 else None
    base = sanitize_text(spec.y_label) or ("Wert" if spec.language == "de" else "Value")
    return f"{base} ({unit})" if unit and unit.lower() not in base.lower() else base


def _default_title(spec: ChartSpec, x_title: str, y_title: str) -> str:
    """Title used when the LLM gave none."""
    joiner = "nach" if spec.language == "de" else "by"
    return f"{y_title} {joiner} {x_title}"


def _finish(figure: go.Figure, language: Language) -> None:
    """Common layout: margins, legend, and the data source under the plot."""
    figure.update_layout(
        margin={"l": 60, "r": 20, "t": 60, "b": 90},
        legend_title_text="",
        template="plotly_white",
    )
    figure.add_annotation(
        text=SOURCE_NOTES[language],
        xref="paper",
        yref="paper",
        x=0,
        y=-0.22,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font={"size": 10, "color": "#666"},
    )


def _frame(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], kinds: dict[str, str]
) -> pd.DataFrame:
    """Result rows as a DataFrame with real datetimes and floats."""
    frame = pd.DataFrame([list(row) for row in rows], columns=list(columns))
    for column, kind in kinds.items():
        if kind == "time":
            frame[column] = pd.to_datetime(frame[column])
        elif kind == "number":
            frame[column] = frame[column].astype(float)
    return frame


def _unit(column: str) -> str | None:
    """Unit implied by a column name, if any."""
    name = column.lower()
    for suffix, unit in UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return None


def _humanize(column: str) -> str:
    """'avg_price_eur_per_mwh' -> 'Average price'."""
    name = column.lower()
    for suffix, _ in UNIT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    words = [WORD_NAMES.get(word, word) for word in name.split("_") if word]
    text = " ".join(words) or column
    return text[:1].upper() + text[1:]


def _kind(values: list[Any]) -> str:
    """Classify a list of non-null values."""
    if not values:
        return "empty"
    if all(isinstance(v, datetime | date) or _is_iso_date(v) for v in values):
        return "time"
    if all(isinstance(v, int | float | Decimal) and not isinstance(v, bool) for v in values):
        return "number"
    return "category"


def _is_iso_date(value: Any) -> bool:
    """True for ISO date/datetime strings, as produced by QueryResult.to_dict()."""
    if not isinstance(value, str) or len(value) < 10 or value[4] != "-":
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _distinct(columns: Sequence[str], rows: Sequence[Sequence[Any]], column: str) -> int:
    """Number of distinct values in a column."""
    index = list(columns).index(column)
    return len({row[index] for row in rows})


def _numbers(columns: Sequence[str], rows: Sequence[Sequence[Any]], column: str) -> list[float]:
    """Numeric values of a column."""
    index = list(columns).index(column)
    return [float(row[index]) for row in rows if row[index] is not None]
