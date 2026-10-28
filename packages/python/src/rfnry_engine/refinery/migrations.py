import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NamedTuple

from rfnry_engine.errors import ConfigurationError, InputError, StorageError


class Call(NamedTuple):
    name: str
    run: Callable[[Any], Awaitable[None]]


class Migration(NamedTuple):
    version: int
    name: str
    statements: Sequence[str | Call]


class Plan(NamedTuple):
    steps: list[str]
    statements: list[str | Call]


Planner = Callable[[Any], Awaitable[Plan]]


_CREATE = (
    """CREATE TABLE IF NOT EXISTS engine_migrations (
        version INTEGER NOT NULL PRIMARY KEY, name TEXT NOT NULL,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS engine_runs (
        scope TEXT NOT NULL, run_id TEXT NOT NULL, parent_run_id TEXT NULL,
        agent TEXT NOT NULL, kind TEXT NOT NULL, task TEXT NULL,
        at TIMESTAMPTZ NOT NULL, recorded TIMESTAMPTZ NOT NULL,
        duration_ms INTEGER NOT NULL, model TEXT NULL, body JSON NOT NULL,
        PRIMARY KEY (scope, run_id))""",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_at ON engine_runs (scope, at)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_recorded ON engine_runs (scope, recorded)",
    """CREATE TABLE IF NOT EXISTS engine_labels (
        scope TEXT NOT NULL, label_id TEXT NOT NULL, run_id TEXT NOT NULL,
        source TEXT NOT NULL, at TIMESTAMPTZ NOT NULL, body JSON NOT NULL,
        PRIMARY KEY (scope, label_id))""",
    "CREATE INDEX IF NOT EXISTS ix_engine_labels_scope_run ON engine_labels (scope, run_id)",
)


async def _fill_tags(conn: Any) -> None:
    from rfnry_engine.refinery.postgres import stored_tags

    after: tuple[str, str] = ("", "")
    while True:
        rows = await conn.fetch(
            "SELECT scope, run_id, body::text AS body FROM engine_runs "
            "WHERE (scope, run_id) > ($1, $2) ORDER BY scope, run_id LIMIT 1000",
            *after,
        )
        if not rows:
            return
        await conn.executemany(
            "UPDATE engine_runs SET tags = $3::jsonb WHERE scope = $1 AND run_id = $2",
            [(r["scope"], r["run_id"], stored_tags(json.loads(r["body"]))) for r in rows],
        )
        after = (rows[-1]["scope"], rows[-1]["run_id"])


_TAGS: tuple[str | Call, ...] = (
    "ALTER TABLE engine_runs ADD COLUMN IF NOT EXISTS tags JSONB",
    Call("fill engine_runs.tags from each run's body, computed by rfnry-engine", _fill_tags),
    "ALTER TABLE engine_runs ALTER COLUMN tags SET NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_parent ON engine_runs (scope, parent_run_id)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_tags ON engine_runs USING GIN (tags jsonb_path_ops)",
)


MIGRATIONS = [Migration(1, "create", _CREATE), Migration(2, "tags", _TAGS)]


LATEST = MIGRATIONS[-1].version


LOCK = "SELECT pg_advisory_xact_lock(hashtextextended('engine.migrate:' || current_schema(), 0))"


DEFAULT = "engine"


_INITIALIZED = ("engine_labels", "engine_runs")


_WRITES = ("SELECT", "INSERT", "UPDATE", "DELETE")


_STATE = (
    "SELECT current_schema() AS schema, current_setting('search_path') AS path, "
    "current_user AS role, "
    "to_regclass(quote_ident(current_schema()) || '.engine_migrations') IS NOT NULL AS migrated, "
    "to_regclass(quote_ident(current_schema()) || '.engine_runs') IS NOT NULL AS tables"
)


_NAMES = re.compile(r'"((?:[^"]|"")*)"|([^,\s]+)')


_ELSEWHERE = (
    "SELECT n.nspname AS name FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE c.relname = 'engine_runs' AND c.relkind IN ('r', 'p') AND n.nspname <> $1 "
    "AND NOT EXISTS (SELECT 1 FROM pg_class m WHERE m.relnamespace = n.oid "
    "AND m.relname = 'engine_migrations') ORDER BY 1"
)


_OCCUPIED = (
    "SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE n.nspname = $1 AND c.relkind IN ('r', 'p'))"
)


_MIGRATED = "SELECT to_regclass(quote_ident($1) || '.engine_migrations') IS NOT NULL"


_LACKING = (
    "SELECT t.name AS table_name, p.name AS privilege "
    "FROM unnest($1::text[]) WITH ORDINALITY AS t(name, n) "
    "CROSS JOIN unnest($2::text[]) WITH ORDINALITY AS p(name, m) "
    "WHERE NOT has_table_privilege(quote_ident(current_schema()) || '.' || t.name, p.name) "
    "ORDER BY p.m, t.name"
)


def _names(path: str, role: str, schema: str | None) -> list[str]:
    found: list[str] = []
    for quoted, bare in _NAMES.findall(path):
        name: str = quoted.replace('""', '"') if quoted else bare.lower()
        if name == "$user":
            if role == schema:
                found.append(role)
            continue
        if name:
            found.append(name)
    return found


def _first(path: str, role: str, schema: str | None) -> str | None:
    names = _names(path, role, schema)
    return names[0] if names else None


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _recorded(migration: Migration) -> str:
    return (
        "INSERT INTO engine_migrations (version, name) "
        f"VALUES ({migration.version}, {_literal(migration.name)})"
    )


def _steps(migrations: Sequence[Migration]) -> Plan:
    return Plan(
        [f"{m.version:04d} {m.name}" for m in migrations],
        [s for m in migrations for s in [*m.statements, _recorded(m)]],
    )


def _shared(schema: str) -> bool:
    return schema in ("public", "information_schema") or schema.startswith("pg_")


async def remedy(conn: Any, schema: str, path: Sequence[str] = ()) -> str:
    moved = _shared(schema) or bool(await conn.fetchval(_OCCUPIED, schema))
    target = DEFAULT if moved else schema
    tail = " and put it first on the pool's search_path" if moved else ""
    if moved and await conn.fetchval(_MIGRATED, target):
        return f"engine's tables are in {target} — put it first on the pool's search_path"
    found = [row["name"] for row in await conn.fetch(_ELSEWHERE, target)]
    near = [name for name in found if _shared(name) or name in path]
    if near:
        return (
            f"the engine tables initialize() made are in {near[0]} — "
            f"run rfnry-engine upgrade --schema {target} --from {near[0]}{tail}"
        )
    if found:
        return (
            f"run rfnry-engine create --schema {target}{tail}; if the tables initialize() made "
            f"in {', '.join(found)} are this app's, run rfnry-engine upgrade --schema {target} "
            "--from that schema instead"
        )
    return f"run rfnry-engine create --schema {target}{tail}"


async def _lacking(conn: Any, role: str) -> list[str]:
    rows = [
        *await conn.fetch(_LACKING, list(_INITIALIZED), list(_WRITES)),
        *await conn.fetch(_LACKING, ["engine_migrations"], ["SELECT"]),
    ]
    if not rows:
        return []
    privileges = list(dict.fromkeys(row["privilege"] for row in rows))
    tables = sorted({row["table_name"] for row in rows})
    return [
        f"{role} lacks {', '.join(privileges)} on {' and '.join(tables)} — grant it "
        f"{', '.join(_WRITES)} on them and SELECT on engine_migrations"
    ]


async def problems(conn: Any) -> list[str]:
    state = await conn.fetchrow(_STATE)
    schema, path, role = state["schema"], state["path"], state["role"]
    names = _names(path, role, schema)
    first = names[0] if names else None
    if first is not None and first != schema:
        held = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)", first
        )
        if held:
            return [
                f"{role} cannot use schema {first} — grant it USAGE on the schema and SELECT, "
                "INSERT, UPDATE, DELETE on its tables"
            ]
        return [
            f"there is no schema {first} on search_path {path!r} — "
            f"{await remedy(conn, first, names)}"
        ]
    if schema is None:
        return [f"no schema on search_path {path!r} exists — {await remedy(conn, DEFAULT, names)}"]
    if not state["migrated"]:
        if state["tables"] and _shared(schema):
            return [
                f"the engine tables in {schema} were made by initialize() — run rfnry-engine "
                f"upgrade --from {schema} and put the engine schema first on the pool's search_path"
            ]
        if state["tables"]:
            return [
                f"the engine tables in {schema} were made by initialize() — "
                f"run rfnry-engine upgrade --schema {schema}"
            ]
        if _shared(schema):
            return [
                f"no engine schema on search_path {path!r} — {await remedy(conn, schema, names)}"
            ]
        return [f"no engine schema in {schema} — {await remedy(conn, schema, names)}"]
    lacking = await _lacking(conn, role)
    if lacking:
        return lacking
    version = int(await conn.fetchval("SELECT max(version) FROM engine_migrations") or 0)
    if version < LATEST:
        return [
            f"{schema} is at version {version}, this rfnry-engine needs {LATEST} — "
            f"run rfnry-engine upgrade --schema {schema}"
        ]
    if version > LATEST:
        return [_ahead(schema, version)]
    return []


def _ahead(schema: str, version: int) -> str:
    return (
        f"{schema} is at version {version}, this rfnry-engine knows {LATEST} — upgrade rfnry-engine"
    )


async def plan_create(conn: Any, *, fresh: bool = False) -> Plan:
    state = await conn.fetchrow(_STATE)
    schema = state["schema"]
    if schema is None or _shared(schema):
        raise InputError(
            f"refinery/migrations: engine is not created in {schema!r} — name a schema of its own"
        )
    if state["tables"] or state["migrated"]:
        raise ConfigurationError(
            f"refinery/migrations: {schema} already holds engine tables — run rfnry-engine "
            "upgrade, or create into another schema"
        )
    found = [row["name"] for row in await conn.fetch(_ELSEWHERE, schema)]
    shared = [name for name in found if _shared(name)]
    if shared and not fresh:
        raise ConfigurationError(
            f"refinery/migrations: the engine tables initialize() made are in {shared[0]} — run "
            f"rfnry-engine upgrade --schema {schema} --from {shared[0]} to keep their runs, or "
            "create with --fresh to start without them"
        )
    return _steps(MIGRATIONS)


async def plan_upgrade(conn: Any, *, source: str | None = None) -> Plan:
    state = await conn.fetchrow(_STATE)
    schema = state["schema"]
    moves = await _moves(conn, state, source) if source is not None else Plan([], [])
    if not state["migrated"] and not (state["tables"] or moves.steps):
        raise ConfigurationError(
            f"refinery/migrations: no engine schema in {schema} — {await remedy(conn, schema)}"
        )
    if not state["migrated"]:
        rest = _steps(MIGRATIONS[1:])
        return Plan(
            [*moves.steps, "0001 create, over the tables initialize() made", *rest.steps],
            [
                *moves.statements,
                *MIGRATIONS[0].statements,
                _recorded(MIGRATIONS[0]),
                *rest.statements,
            ],
        )
    rows = await conn.fetch("SELECT version FROM engine_migrations ORDER BY version")
    applied = [int(row["version"]) for row in rows]
    newest = max(applied, default=0)
    if newest > LATEST:
        raise ConfigurationError(f"refinery/migrations: {_ahead(schema, newest)}")
    pending = [m for m in MIGRATIONS if m.version not in applied]
    return _steps(pending) if pending else Plan([], [])


async def _moves(conn: Any, state: Any, source: str) -> Plan:
    held = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = $1 AND tablename = ANY($2::text[]) "
        "ORDER BY tablename",
        source,
        list(_INITIALIZED),
    )
    names = [row["tablename"] for row in held]
    if "engine_runs" not in names and state["migrated"]:
        return Plan([], [])
    if "engine_runs" not in names:
        raise ConfigurationError(
            f"refinery/migrations: {source} holds no engine tables initialize() made"
        )
    migrated = await conn.fetchval(
        "SELECT to_regclass(quote_ident($1) || '.engine_migrations') IS NOT NULL", source
    )
    if migrated:
        raise ConfigurationError(
            f"refinery/migrations: {source} is an engine schema of its own — point the pool at it"
        )
    if state["tables"] or state["migrated"]:
        raise ConfigurationError(
            f"refinery/migrations: {state['schema']} already holds engine tables — "
            "move them into a schema of their own"
        )
    target = await conn.fetchval("SELECT quote_ident(current_schema())")
    origin = await conn.fetchval("SELECT quote_ident($1)", source)
    return Plan(
        [f"move {', '.join(names)} from {source}"],
        [f"ALTER TABLE {origin}.{name} SET SCHEMA {target}" for name in names],
    )


async def apply(
    conn: Any,
    planner: Planner,
    *,
    run: bool = True,
    expected: Plan | None = None,
    before: Sequence[str] = (),
) -> Plan:
    transaction = conn.transaction(isolation="read_committed")
    await transaction.start()
    try:
        for made in before:
            await conn.execute(made)
        await conn.execute(LOCK)
        planned = await planner(conn)
        plan = Plan(planned.steps, [*before, *planned.statements])
        if expected is not None and plan != expected:
            raise ConfigurationError(
                "refinery/migrations: the schema changed since the plan was shown — "
                "run the command again"
            )
        if run:
            for statement in planned.statements:
                await _execute(conn, statement)
    except BaseException:
        await transaction.rollback()
        raise
    if run:
        await transaction.commit()
    else:
        await transaction.rollback()
    return plan


async def _execute(conn: Any, statement: str | Call) -> None:
    if isinstance(statement, Call):
        try:
            await statement.run(conn)
        except Exception as exc:
            raise StorageError(f"refinery/migrations: {statement.name} failed: {exc}") from exc
        return
    try:
        await conn.execute(statement)
    except Exception as exc:
        raise StorageError(
            f"refinery/migrations: {statement.split(chr(10))[0]} failed: {exc}"
        ) from exc
