"""Static validation of LLM-generated SQL with sqlglot.

This is the first safety layer: it rejects anything that is not a single,
read-only SELECT over the known GridQuery tables, before DuckDB ever sees it.
The database connection enforces the same rules again (see sql_executor), so a
gap in one layer is caught by the other.
"""

from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

DIALECT = "duckdb"
MAX_SQL_LENGTH = 10_000

# Statement and clause types that change data, schema, session state or files.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Copy,
    exp.Attach,
    exp.Detach,
    exp.Install,
    exp.Pragma,
    exp.Set,
    exp.Use,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Command,
)

# Functions that may be used as a table source in FROM.
ALLOWED_TABLE_FUNCTIONS = frozenset({"generate_series", "range", "unnest"})

# Scalar/table functions that read files, URLs, environment, settings or run dynamic SQL.
FORBIDDEN_FUNCTIONS = frozenset(
    {"getenv", "query", "query_table", "glob", "current_setting", "sql_auto_complete"}
)
FORBIDDEN_FUNCTION_PREFIXES = (
    "read_",
    "duckdb_",
    "pragma_",
    "sqlite_",
    "postgres_",
    "mysql_",
    "parquet_",
    "iceberg_",
    "delta_",
    "http",
    "st_read",
)

ALLOWED_SCHEMAS = frozenset({"", "main"})


class SqlValidationError(ValueError):
    """Raised when a query is rejected; the message is safe to show to the LLM."""


@dataclass(frozen=True)
class ValidatedQuery:
    """A query that passed validation.

    Attributes:
        original_sql: The SQL as submitted.
        sql: The query regenerated from the validated syntax tree. This is what gets
            executed, so DuckDB runs exactly what was checked.
        tables: Database tables and views the query reads.
    """

    original_sql: str
    sql: str
    tables: frozenset[str]


def validate_sql(sql: str, allowed_tables: Collection[str]) -> ValidatedQuery:
    """Check that `sql` is a single read-only SELECT over allowed tables.

    Args:
        sql: SQL text proposed by the agent.
        allowed_tables: Names of tables and views the agent may read.

    Returns:
        The validated query.

    Raises:
        SqlValidationError: If the query is not allowed. The message explains why.
    """
    statement = _parse_single_statement(sql)
    if not isinstance(statement, exp.Query):
        if isinstance(statement, (*FORBIDDEN_NODES, exp.DDL, exp.DML, exp.Describe)):
            found = _describe(statement)
        else:  # e.g. a misspelled keyword that sqlglot read as an expression
            found = "something that is not a valid statement (check the syntax)"
        raise SqlValidationError(f"Only read-only SELECT queries are allowed, got {found}.")

    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlValidationError(f"{_describe(node)} is not allowed; only SELECT is.")

    _check_functions(statement)
    tables = _check_tables(statement, {name.lower() for name in allowed_tables})
    return ValidatedQuery(sql, statement.sql(dialect=DIALECT), tables)


def _parse_single_statement(sql: str) -> exp.Expression:
    """Parse the SQL and require exactly one statement."""
    if not sql or not sql.strip():
        raise SqlValidationError("The query is empty.")
    if len(sql) > MAX_SQL_LENGTH:
        raise SqlValidationError(f"The query is too long (max {MAX_SQL_LENGTH} characters).")
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except (ParseError, TokenError) as exc:
        raise SqlValidationError(f"SQL syntax error: {_first_line(exc)}") from exc
    if len(statements) != 1:
        raise SqlValidationError(f"Exactly one SQL statement is allowed, found {len(statements)}.")
    return statements[0]


def _check_functions(statement: exp.Expression) -> None:
    """Reject functions that reach outside the database or run dynamic SQL."""
    for func in statement.find_all(exp.Func):
        name = _function_name(func)
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            raise SqlValidationError(f"Function {name}() is not allowed.")


def _check_tables(statement: exp.Expression, allowed: set[str]) -> frozenset[str]:
    """Require every table reference to be an allowed table, a CTE, or a safe function."""
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    used = set()
    for table in statement.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            name = _function_name(table.this)
            if name not in ALLOWED_TABLE_FUNCTIONS:
                raise SqlValidationError(f"Table function {name}() is not allowed.")
            continue
        name = table.name.lower()
        if table.catalog or table.db.lower() not in ALLOWED_SCHEMAS:
            raise SqlValidationError(
                f"Table {table.sql(dialect=DIALECT)} is not allowed; "
                "use unqualified table names from the GridQuery schema."
            )
        if name in cte_names and not table.db:
            continue
        if name not in allowed:
            raise SqlValidationError(
                f"Unknown table {table.name!r}. Available tables: {', '.join(sorted(allowed))}."
            )
        used.add(name)
    return frozenset(used)


def _function_name(func: exp.Func) -> str:
    """Return a function's SQL name in lower case."""
    return (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()


def _describe(node: exp.Expression) -> str:
    """Name a statement or clause for error messages, e.g. 'DROP' or 'INTO'."""
    if isinstance(node, exp.Command):
        return str(node.this).upper()
    return node.key.upper()


def _first_line(exc: Exception) -> str:
    """Return the first line of an exception message."""
    return str(exc).strip().splitlines()[0]
