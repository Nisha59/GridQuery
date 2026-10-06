"""Compact description of the database for the agent's prompt.

Combines what DuckDB knows (tables, columns, types, date ranges, example rows,
energy source metadata) with the hand-written texts in descriptions.toml, and
renders it as short plain text. Plain text is used instead of JSON because it
needs far fewer tokens for the same information.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import duckdb

from gridquery.database.descriptions import Descriptions, load_descriptions
from gridquery.tools.sql_executor import SafeSqlExecutor

logger = logging.getLogger(__name__)

# Columns that define a relation's time axis, in order of preference.
TIME_COLUMNS = ("timestamp_utc", "date_local", "month_local")


@dataclass(frozen=True)
class ColumnInfo:
    """One column as the agent sees it."""

    name: str
    data_type: str
    unit: str | None
    description: str | None


@dataclass(frozen=True)
class RelationInfo:
    """One table or view as the agent sees it."""

    name: str
    kind: str  # "table" or "view"
    description: str | None
    columns: list[ColumnInfo]
    row_count: int
    time_column: str | None
    first: Any
    last: Any
    examples: list[tuple[Any, ...]]


@dataclass(frozen=True)
class SourceInfo:
    """One energy source with its data range."""

    source: str
    name_en: str
    name_de: str
    is_renewable: bool
    is_fossil: bool
    description: str
    first_utc: datetime | None
    last_utc: datetime | None


@dataclass(frozen=True)
class SchemaContext:
    """Everything the agent needs to know about the database."""

    description: str
    notes: list[str]
    relations: list[RelationInfo]
    sources: list[SourceInfo]


def build_schema_context(
    con: duckdb.DuckDBPyConnection,
    descriptions: Descriptions,
    *,
    example_rows: int = 2,
) -> SchemaContext:
    """Collect schema facts from DuckDB and merge them with the descriptions.

    Args:
        con: Connection to the GridQuery database (only fixed internal queries are run).
        descriptions: Hand-written documentation.
        example_rows: Example rows per table; views get at most one.
    """
    kinds = _relation_kinds(con)
    columns = _columns(con)
    # Documented relations first, in the curated order of the TOML file.
    names = [n for n in descriptions.relations if n in kinds]
    names += sorted(n for n in kinds if n not in descriptions.relations)
    for name in names:
        if name not in descriptions.relations:
            logger.warning("Relation %s has no description in descriptions.toml", name)

    relations = [
        _relation_info(con, name, kinds[name], columns[name], descriptions, example_rows)
        for name in names
    ]
    return SchemaContext(descriptions.description, descriptions.notes, relations, _sources(con))


def render_schema_context(context: SchemaContext) -> str:
    """Render the context as compact plain text for the LLM prompt."""
    lines = [f"Database (DuckDB SQL): {context.description}", "", "Notes:"]
    lines += [f"- {note}" for note in context.notes]
    lines += ["", "Tables and views (column TYPE [unit]: description):"]
    rendered: list[RelationInfo] = []
    for relation in context.relations:
        same_as = next((r for r in rendered if _same_measures(r, relation)), None)
        lines += ["", *_render_relation(relation, same_as)]
        rendered.append(relation)
    lines += ["", *_render_sources(context.sources)]
    return "\n".join(lines)


def find_undocumented(con: duckdb.DuckDBPyConnection, descriptions: Descriptions) -> list[str]:
    """List mismatches between the database schema and the descriptions file."""
    columns = _columns(con)
    problems = []
    for name, cols in columns.items():
        documented = descriptions.relations.get(name)
        if documented is None:
            problems.append(f"{name}: missing from descriptions")
            continue
        actual = {c for c, _ in cols}
        problems += [
            f"{name}.{c}: missing from descriptions" for c in actual - documented.columns.keys()
        ]
        problems += [f"{name}.{c}: not in database" for c in documented.columns.keys() - actual]
    problems += [f"{n}: not in database" for n in descriptions.relations.keys() - columns.keys()]
    return sorted(problems)


class SchemaContextProvider:
    """Builds the schema context once per database and serves it as text."""

    def __init__(
        self,
        executor: SafeSqlExecutor,
        descriptions: Descriptions | None = None,
        *,
        example_rows: int = 2,
    ) -> None:
        """Create a provider that reads through the executor's hardened connection."""
        self._executor = executor
        self._descriptions = descriptions or load_descriptions()
        self._example_rows = example_rows
        self._text: str | None = None

    def text(self) -> str:
        """Return the rendered schema context, building it on first use."""
        if self._text is None:
            with self._executor.internal_cursor() as cursor:
                context = build_schema_context(
                    cursor, self._descriptions, example_rows=self._example_rows
                )
            self._text = render_schema_context(context)
            logger.info("Schema context built: %d characters", len(self._text))
        return self._text


# --- Introspection (fixed internal queries only) ----------------------------------------------


def _relation_kinds(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Map each user table/view name to 'table' or 'view'."""
    return dict(
        con.execute(
            "SELECT table_name, 'table' FROM duckdb_tables() WHERE NOT internal "
            "UNION ALL SELECT view_name, 'view' FROM duckdb_views() WHERE NOT internal"
        ).fetchall()
    )


def _columns(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[str, str]]]:
    """Map each relation to its (column, type) pairs in order."""
    rows = con.execute(
        "SELECT table_name, list((column_name, data_type) ORDER BY column_index) "
        "FROM duckdb_columns() WHERE NOT internal GROUP BY table_name"
    ).fetchall()
    return {name: [tuple(col) for col in cols] for name, cols in rows}


def _relation_info(
    con: duckdb.DuckDBPyConnection,
    name: str,
    kind: str,
    columns: list[tuple[str, str]],
    descriptions: Descriptions,
    example_rows: int,
) -> RelationInfo:
    """Gather columns, size, time range and examples for one relation."""
    documented = descriptions.relations.get(name)
    column_docs = documented.columns if documented else {}
    infos = [
        ColumnInfo(
            column,
            data_type,
            column_docs[column].unit if column in column_docs else None,
            column_docs[column].description if column in column_docs else None,
        )
        for column, data_type in columns
    ]
    names = [column for column, _ in columns]
    time_column = next((c for c in TIME_COLUMNS if c in names), None)
    # Identifiers come from DuckDB's own catalog, not from user input.
    range_sql = f"min({time_column}), max({time_column})" if time_column else "NULL, NULL"
    row_count, first, last = con.execute(f'SELECT count(*), {range_sql} FROM "{name}"').fetchone()
    examples = con.execute(
        f'SELECT * FROM "{name}" WHERE COLUMNS(*) IS NOT NULL ORDER BY ALL DESC LIMIT ?',
        [example_rows if kind == "table" else min(example_rows, 1)],
    ).fetchall()
    return RelationInfo(
        name=name,
        kind=kind,
        description=documented.description if documented else None,
        columns=infos,
        row_count=row_count,
        time_column=time_column,
        first=first,
        last=last,
        examples=examples,
    )


def _sources(con: duckdb.DuckDBPyConnection) -> list[SourceInfo]:
    """Read the energy_sources table with each source's data range."""
    rows = con.execute("""
        SELECT s.source, s.name_en, s.name_de, s.is_renewable, s.is_fossil, s.description,
               min(g.timestamp_utc), max(g.timestamp_utc)
        FROM energy_sources AS s LEFT JOIN generation AS g USING (source)
        GROUP BY ALL ORDER BY s.is_renewable, s.is_fossil DESC, s.source
    """).fetchall()
    return [SourceInfo(*row) for row in rows]


# --- Rendering ---------------------------------------------------------------------------------


def _same_measures(first: RelationInfo, second: RelationInfo) -> bool:
    """True if two relations differ only in their first (time) column."""
    return len(first.columns) == len(second.columns) > 1 and first.columns[1:] == second.columns[1:]


def _render_relation(relation: RelationInfo, same_as: RelationInfo | None = None) -> list[str]:
    """Render one table/view block; columns shared with `same_as` are not repeated."""
    facts = [relation.kind]
    if relation.kind == "table":
        facts.append(f"{relation.row_count:,} rows")
    if relation.time_column and relation.first is not None:
        suffix = " UTC" if relation.time_column.endswith("_utc") else ""
        facts.append(f"{_fmt(relation.first)} to {_fmt(relation.last)}{suffix}")
    header = f"{relation.name} ({'; '.join(facts)})"
    if relation.description:
        header += f": {relation.description}"

    lines = [header]
    columns = relation.columns[:1] if same_as else relation.columns
    for column in columns:
        unit = f" [{column.unit}]" if column.unit else ""
        text = f": {column.description}" if column.description else ""
        lines.append(f"  {column.name} {column.data_type}{unit}{text}")
    if same_as:
        lines.append(f"  (other columns as in {same_as.name})")
    if relation.examples and relation.name != "energy_sources":
        lines.append("  e.g. " + " ; ".join(_fmt_row(row) for row in relation.examples))
    return lines


def _render_sources(sources: list[SourceInfo]) -> list[str]:
    """Render all energy sources; flag sources whose data range is unusual."""
    lines = ["Energy sources (source | English | German | class | notes):"]
    latest = max((s.last_utc for s in sources if s.last_utc), default=None)
    for s in sources:
        klass = "renewable" if s.is_renewable else "fossil" if s.is_fossil else "other"
        note = s.description
        if s.last_utc is None:
            note += " No data."
        elif latest and s.last_utc < latest:
            note += f" Data only until {_fmt(s.last_utc)} UTC."
        lines.append(f"  {s.source} | {s.name_en} | {s.name_de} | {klass} | {note}")
    return lines


def _fmt_row(row: tuple[Any, ...]) -> str:
    """Format an example row compactly."""
    return "(" + ", ".join(_fmt(value) for value in row) + ")"


def _fmt(value: Any) -> str:
    """Format one value compactly: short timestamps, rounded floats."""
    if value is None:
        return "NULL"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1 else f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)
