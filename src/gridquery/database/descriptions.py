"""Load table and column descriptions from descriptions.toml.

The TOML file is the single source of truth for documentation: the loader stores
it as DuckDB comments, and the schema context tool shows it to the agent.
"""

import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class ColumnDescription:
    """Documentation of one column."""

    description: str
    unit: str | None = None


@dataclass(frozen=True)
class RelationDescription:
    """Documentation of one table or view."""

    description: str
    columns: dict[str, ColumnDescription] = field(default_factory=dict)


@dataclass(frozen=True)
class Descriptions:
    """Documentation of the whole database."""

    description: str
    notes: list[str]
    relations: dict[str, RelationDescription]


def load_descriptions(path: Path | None = None) -> Descriptions:
    """Parse descriptions.toml (or another file with the same layout).

    Args:
        path: File to read; defaults to the packaged descriptions.toml.
    """
    if path is None:
        text = files("gridquery.database").joinpath("descriptions.toml").read_text("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    relations = {
        name: RelationDescription(
            description=relation["description"],
            columns={
                column: ColumnDescription(spec["description"], spec.get("unit"))
                for column, spec in relation.get("columns", {}).items()
            },
        )
        for name, relation in data["relations"].items()
    }
    database = data["database"]
    return Descriptions(database["description"], list(database.get("notes", [])), relations)


def apply_comments(con: duckdb.DuckDBPyConnection, descriptions: Descriptions) -> None:
    """Store the descriptions as DuckDB COMMENTs on tables, views and columns."""
    kinds = dict(
        con.execute(
            "SELECT table_name, 'TABLE' FROM duckdb_tables() WHERE NOT internal "
            "UNION ALL SELECT view_name, 'VIEW' FROM duckdb_views() WHERE NOT internal"
        ).fetchall()
    )
    for name, relation in descriptions.relations.items():
        if name not in kinds:
            continue
        con.execute(f"COMMENT ON {kinds[name]} {name} IS {_quote(relation.description)}")
        for column, spec in relation.columns.items():
            text = f"{spec.description} [{spec.unit}]" if spec.unit else spec.description
            con.execute(f"COMMENT ON COLUMN {name}.{column} IS {_quote(text)}")


def _quote(text: str) -> str:
    """Quote a string literal for SQL."""
    return "'" + text.replace("'", "''") + "'"
