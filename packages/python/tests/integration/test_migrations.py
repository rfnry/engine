import asyncio
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from database import URL, Schemas, admin, quoted
from unit.fakes import Fixed, MemoryBucket, Scripted, generation_record

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, StorageError
from rfnry_engine.models import CheckAnswer, Message, Reply, ScoreAnswer, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery import migrations
from rfnry_engine.refinery.labels import Label, Review
from rfnry_engine.refinery.migrations import Migration, Plan
from rfnry_engine.refinery.postgres import PostgresStorage, stored_tags
from rfnry_engine.refinery.refinery import Refinery, flatten, label_of, read_back
from rfnry_engine.refinery.storage import RunRow

BEFORE_TAGS = [
    """
    CREATE TABLE IF NOT EXISTS engine_runs (
        scope TEXT NOT NULL, run_id TEXT NOT NULL, parent_run_id TEXT NULL,
        agent TEXT NOT NULL, kind TEXT NOT NULL, task TEXT NULL,
        at TIMESTAMPTZ NOT NULL, recorded TIMESTAMPTZ NOT NULL,
        duration_ms INTEGER NOT NULL, model TEXT NULL, body JSON NOT NULL,
        PRIMARY KEY (scope, run_id))
    """,
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_at ON engine_runs (scope, at)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_recorded ON engine_runs (scope, recorded)",
    """
    CREATE TABLE IF NOT EXISTS engine_labels (
        scope TEXT NOT NULL, label_id TEXT NOT NULL, run_id TEXT NOT NULL,
        source TEXT NOT NULL, at TIMESTAMPTZ NOT NULL, body JSON NOT NULL,
        PRIMARY KEY (scope, label_id))
    """,
    "CREATE INDEX IF NOT EXISTS ix_engine_labels_scope_run ON engine_labels (scope, run_id)",
]
WITH_TAGS = [
    BEFORE_TAGS[0].replace("body JSON NOT NULL,", "body JSON NOT NULL, tags JSONB NOT NULL,"),
    *BEFORE_TAGS[1:3],
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_parent ON engine_runs (scope, parent_run_id)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_tags ON engine_runs USING GIN (tags jsonb_path_ops)",
    *BEFORE_TAGS[3:],
]
TABLES = ["engine_labels", "engine_runs"]
AT = datetime(2026, 1, 1, 12, tzinfo=UTC)
NOW = migrations.LATEST
NEXT = NOW + 1
PROBE = Migration(NEXT, "probe", ("CREATE TABLE engine_probe (n INTEGER)",))


async def created(schemas: Schemas) -> tuple[str, Any]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    async with pool.acquire() as conn:
        await migrations.apply(conn, migrations.plan_create)
    return name, pool


async def initialized(schemas: Schemas, ddl: list[str] = WITH_TAGS) -> tuple[str, Any]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    async with pool.acquire() as conn:
        for statement in ddl:
            await conn.execute(statement)
    return name, pool


def run(run_id: str, tags: dict[str, str] | None = None, reply: str = "") -> RunRow:
    record = generation_record(agent="lead", reply=reply or run_id, run_id=run_id)
    [row] = flatten(record.model_copy(update={"tags": tags or {}, "at": AT}), AT)
    return row


APPROVED = Label(
    label_id="l1",
    run_id="r1",
    scope="acme",
    at=AT,
    source="human",
    review=Review(stance="approval"),
).model_dump(mode="json")


HOSTILE = run("r3", {"customer": "c-퀴"}, reply="tool said a\x00b \udc80")
OLD_TAGS = "replace(replace($10::text, '\\ud', '\\u0'), '\\u0000', '\\ufffd')::jsonb"


async def written_before(pool: Any, tagged: bool) -> None:
    async with pool.acquire() as conn:
        for row in (run("r1", {"customer": "c-81"}), run("r2"), HOSTILE):
            columns = "scope, run_id, agent, kind, at, recorded, duration_ms, model, body"
            values = "$1, $2, $3, $4, $5, $6, $7, $8, $9::json"
            args = [row.scope, row.run_id, row.agent, row.kind, row.at, row.recorded, 1, "m"]
            args.append(json.dumps(row.body))
            if tagged:
                columns += ", tags"
                values += f", {OLD_TAGS}"
                args.append(json.dumps(row.body.get("tags") or {}))
            await conn.execute(f"INSERT INTO engine_runs ({columns}) VALUES ({values})", *args)
        await conn.execute(
            "INSERT INTO engine_labels (scope, label_id, run_id, source, at, body) "
            "VALUES ('acme', 'l1', 'r1', 'human', $1, $2::text::json)",
            AT,
            json.dumps(APPROVED),
        )


async def kept(storage: PostgresStorage) -> None:
    assert sorted(r.run_id for r in await storage.runs("acme")) == ["r1", "r2", "r3"]
    assert [r.run_id for r in await storage.run_tree("acme", tags={"customer": "c-81"})] == ["r1"]
    assert [r.run_id for r in await storage.run_tree("acme", tags={"customer": "c-퀴"})] == ["r3"]
    assert await storage.run_tree("acme", tags={"customer": "c-4"}) == []
    stored = await storage.get_run("acme", "r3")
    assert stored is not None and stored.body == HOSTILE.body
    assert [label.label_id for label in await storage.labels("acme")] == ["l1"]


async def tables(pool: Any, schema: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = $1 ORDER BY 1", schema
        )
    return [row["tablename"] for row in rows]


def probing(monkeypatch: pytest.MonkeyPatch, *steps: Migration) -> None:
    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, *steps])
    monkeypatch.setattr(migrations, "LATEST", steps[-1].version)


async def test_create_makes_every_table_and_index_and_records_every_version(
    schemas: Schemas,
) -> None:
    name, pool = await created(schemas)
    assert await tables(pool, name) == ["engine_labels", "engine_migrations", "engine_runs"]
    async with pool.acquire() as conn:
        versions = await conn.fetch("SELECT version, name FROM engine_migrations ORDER BY 1")
        indexes = {
            r["indexname"]
            for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = $1", name
            )
        }
    assert [tuple(row) for row in versions] == [(m.version, m.name) for m in migrations.MIGRATIONS]
    assert {"ix_engine_runs_scope_parent", "ix_engine_runs_tags"} <= indexes
    storage = PostgresStorage(pool)
    await storage.check()
    await Refinery(storage).check()
    await storage.put_runs("acme", [run("r1", {"customer": "c-81"}), run("r2")])
    assert [r.run_id for r in await storage.run_tree("acme", tags={"customer": "c-81"})] == ["r1"]


async def test_create_refuses_a_schema_that_already_holds_engine_tables(schemas: Schemas) -> None:
    name, pool = await created(schemas)
    async with pool.acquire() as conn:
        with pytest.raises(
            ConfigurationError,
            match=f"^refinery/migrations: {name} already holds engine tables — run rfnry-engine "
            "upgrade, or create into another schema$",
        ):
            await migrations.apply(conn, migrations.plan_create)


async def test_create_refuses_a_shared_schema_and_runs_nothing_there(schemas: Schemas) -> None:
    pool = await schemas.pool("public")
    async with pool.acquire() as conn:
        with pytest.raises(
            InputError,
            match="^refinery/migrations: engine is not created in 'public' — "
            "name a schema of its own$",
        ):
            await migrations.apply(conn, migrations.plan_create)
    assert [t for t in await tables(pool, "public") if t.startswith("engine_")] == []


async def test_a_dry_run_shows_the_plan_and_runs_nothing(schemas: Schemas) -> None:
    name = await schemas.create()
    pool = await schemas.pool(name)
    async with pool.acquire() as conn:
        shown = await migrations.apply(conn, migrations.plan_create, run=False)
    assert shown.steps == [f"{m.version:04d} {m.name}" for m in migrations.MIGRATIONS]
    assert await tables(pool, name) == []


async def test_upgrade_with_nothing_pending_does_nothing(schemas: Schemas) -> None:
    _, pool = await created(schemas)
    async with pool.acquire() as conn:
        assert await migrations.apply(conn, migrations.plan_upgrade) == Plan([], [])


async def test_upgrade_runs_a_new_migration_once_and_records_it(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, pool = await created(schemas)
    probing(monkeypatch, PROBE)
    async with pool.acquire() as conn:
        ran = await migrations.apply(conn, migrations.plan_upgrade)
        assert ran.steps == [f"{NEXT:04d} probe"]
        assert await migrations.apply(conn, migrations.plan_upgrade) == Plan([], [])
        assert await conn.fetchval("SELECT max(version) FROM engine_migrations") == NEXT
    assert "engine_probe" in await tables(pool, name)
    await PostgresStorage(pool).check()


async def test_a_failing_step_leaves_the_schema_as_it_was(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, pool = await created(schemas)
    broken = Migration(NEXT, "broken", ("CREATE TABLE engine_probe (n INTEGER)", "SELECT 1/0"))
    probing(monkeypatch, broken)
    async with pool.acquire() as conn:
        with pytest.raises(StorageError, match="^refinery/migrations: SELECT 1/0 failed: "):
            await migrations.apply(conn, migrations.plan_upgrade)
        assert await conn.fetchval("SELECT max(version) FROM engine_migrations") == NOW
    assert "engine_probe" not in await tables(pool, name)


async def test_an_error_of_a_step_s_own_code_reaches_the_caller_as_itself(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, pool = await created(schemas)

    async def defect(conn: Any) -> None:
        raise KeyError("tags")

    broken = Migration(
        NEXT,
        "broken",
        ("CREATE TABLE engine_probe (n INTEGER)", migrations.Call("fill engine_probe", defect)),
    )
    probing(monkeypatch, broken)
    async with pool.acquire() as conn:
        with pytest.raises(KeyError, match="tags"):
            await migrations.apply(conn, migrations.plan_upgrade)
        assert await conn.fetchval("SELECT max(version) FROM engine_migrations") == NOW
    assert "engine_probe" not in await tables(pool, name)


async def test_a_step_that_times_out_raises_storage_error_naming_the_timeout(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, pool = await created(schemas)

    async def slow(conn: Any) -> None:
        await conn.execute("SELECT pg_sleep(5)", timeout=0.2)

    waiting = Migration(
        NEXT,
        "waiting",
        ("CREATE TABLE engine_probe (n INTEGER)", migrations.Call("wait on the database", slow)),
    )
    probing(monkeypatch, waiting)
    async with pool.acquire() as conn:
        with pytest.raises(
            StorageError, match="^refinery/migrations: wait on the database failed: TimeoutError$"
        ) as raised:
            await asyncio.wait_for(migrations.apply(conn, migrations.plan_upgrade), 10)
        assert await conn.fetchval("SELECT max(version) FROM engine_migrations") == NOW
    assert "engine_probe" not in await tables(pool, name)
    assert isinstance(raised.value.__cause__, TimeoutError)


async def test_a_plan_the_schema_has_moved_past_is_refused(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool = await created(schemas)
    probing(monkeypatch, PROBE)
    async with pool.acquire() as conn:
        shown = await migrations.apply(conn, migrations.plan_upgrade, run=False)
        await migrations.apply(conn, migrations.plan_upgrade)
        with pytest.raises(
            ConfigurationError,
            match="^refinery/migrations: the schema changed since the plan was shown — "
            "run the command again$",
        ):
            await migrations.apply(conn, migrations.plan_upgrade, expected=shown)


async def test_a_schema_ahead_of_this_version_is_refused(schemas: Schemas) -> None:
    name, pool = await created(schemas)
    async with pool.acquire() as conn:
        await conn.execute(
            f"INSERT INTO engine_migrations (version, name) VALUES ({NEXT}, 'later')"
        )
        ahead = f"{name} is at version {NEXT}, this rfnry-engine knows {NOW} — upgrade rfnry-engine"
        with pytest.raises(ConfigurationError, match=f"^refinery/migrations: {ahead}$"):
            await migrations.apply(conn, migrations.plan_upgrade)
    with pytest.raises(ConfigurationError, match=f"^refinery/postgres: {ahead}$"):
        await PostgresStorage(pool).check()


@pytest.mark.parametrize("ddl", [BEFORE_TAGS, WITH_TAGS], ids=["0.1.3", "0.1.4"])
async def test_upgrade_adopts_the_tables_initialize_made_and_keeps_every_row(
    schemas: Schemas, ddl: list[str]
) -> None:
    name, pool = await initialized(schemas, ddl)
    await written_before(pool, tagged=ddl is WITH_TAGS)
    storage = PostgresStorage(pool)
    with pytest.raises(
        ConfigurationError,
        match=f"^refinery/postgres: the engine tables in {name} were made by initialize\\(\\) — "
        f"run rfnry-engine upgrade --schema {name}$",
    ):
        await storage.check()
    async with pool.acquire() as conn:
        ran = await migrations.apply(conn, migrations.plan_upgrade)
    assert ran.steps == [
        "0001 create, over the tables initialize() made",
        *(f"{m.version:04d} {m.name}" for m in migrations.MIGRATIONS[1:]),
    ]
    await storage.check()
    await kept(storage)
    async with pool.acquire() as conn:
        assert await migrations.apply(conn, migrations.plan_upgrade) == Plan([], [])
        nullable = await conn.fetchval(
            "SELECT is_nullable FROM information_schema.columns WHERE table_schema = $1 "
            "AND table_name = 'engine_runs' AND column_name = 'tags'",
            name,
        )
    assert nullable == "NO"


async def test_upgrade_from_another_schema_moves_the_tables_and_adopts_them(
    schemas: Schemas,
) -> None:
    source, held = await initialized(schemas, BEFORE_TAGS)
    await written_before(held, tagged=False)
    target = schemas.name()
    pool = await schemas.pool(target)
    async with pool.acquire() as conn:
        ran = await migrations.apply(
            conn,
            partial(migrations.plan_upgrade, source=source),
            before=[f"CREATE SCHEMA IF NOT EXISTS {quoted(target)}"],
        )
    assert ran.steps[:2] == [
        f"move {', '.join(TABLES)} from {source}",
        "0001 create, over the tables initialize() made",
    ]
    assert await tables(pool, source) == []
    storage = PostgresStorage(pool)
    await storage.check()
    await kept(storage)


async def test_upgrade_from_refuses_a_source_without_tables_or_a_target_that_holds_them(
    schemas: Schemas,
) -> None:
    empty = await schemas.create()
    fresh = await schemas.create()
    target, pool = await created(schemas)
    source, _ = await initialized(schemas)
    async with (await schemas.pool(fresh)).acquire() as conn:
        with pytest.raises(
            ConfigurationError,
            match=f"^refinery/migrations: {empty} holds no engine tables initialize\\(\\) made$",
        ):
            await migrations.apply(conn, partial(migrations.plan_upgrade, source=empty))
    async with pool.acquire() as conn:
        done = await migrations.apply(conn, partial(migrations.plan_upgrade, source=empty))
        assert done == Plan([], [])
        with pytest.raises(
            ConfigurationError,
            match=f"^refinery/migrations: {target} already holds engine tables — "
            "move them into a schema of their own$",
        ):
            await migrations.apply(conn, partial(migrations.plan_upgrade, source=source))
    assert await tables(pool, source) == TABLES


async def test_upgrade_from_refuses_a_source_the_command_made(schemas: Schemas) -> None:
    source, _ = await created(schemas)
    target = await schemas.create()
    pool = await schemas.pool(target)
    async with pool.acquire() as conn:
        with pytest.raises(
            ConfigurationError,
            match=f"^refinery/migrations: {source} is an engine schema of its own — "
            "point the pool at it$",
        ):
            await migrations.apply(conn, partial(migrations.plan_upgrade, source=source))


async def test_check_names_the_command_for_each_way_a_schema_does_not_fit(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def problem(pool: Any) -> str:
        with pytest.raises(ConfigurationError) as raised:
            await PostgresStorage(pool).check()
        return str(raised.value)

    missing = schemas.name()
    assert await problem(await schemas.pool(missing)) == (
        f"refinery/postgres: there is no schema {missing} on search_path '{quoted(missing)}' — "
        f"run rfnry-engine create --schema {missing}"
    )
    empty = await schemas.create()
    assert await problem(await schemas.pool(empty)) == (
        f"refinery/postgres: no engine schema in {empty} — run rfnry-engine create --schema {empty}"
    )
    assert await problem(await schemas.pool("public")) == (
        "refinery/postgres: no engine schema on search_path '\"public\"' — run rfnry-engine "
        "create --schema engine and put it first on the pool's search_path"
    )
    old, held = await initialized(schemas)
    monkeypatch.setattr(migrations, "_shared", lambda schema: schema == old)
    assert await problem(held) == (
        f"refinery/postgres: the engine tables in {old} were made by initialize() — run "
        f"rfnry-engine upgrade --from {old} and put the engine schema first on the pool's "
        "search_path"
    )
    monkeypatch.undo()
    name, pool = await created(schemas)
    probing(monkeypatch, PROBE)
    assert await problem(pool) == (
        f"refinery/postgres: {name} is at version {NOW}, this rfnry-engine needs {NEXT} — "
        f"run rfnry-engine upgrade --schema {name}"
    )


async def test_check_and_upgrade_point_at_tables_initialize_made_in_a_shared_schema(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, _ = await initialized(schemas)
    monkeypatch.setattr(migrations, "_shared", lambda schema: schema == old)

    async def problem(pool: Any) -> str:
        with pytest.raises(ConfigurationError) as raised:
            await PostgresStorage(pool).check()
        return str(raised.value)

    missing = schemas.name()
    found = await problem(await schemas.pool(missing))
    assert f"there is no schema {missing} on search_path" in found
    assert f"run rfnry-engine upgrade --schema {missing} --from {old}" in found
    empty = await schemas.create()
    pool = await schemas.pool(empty)
    assert f"run rfnry-engine upgrade --schema {empty} --from {old}" in await problem(pool)
    async with pool.acquire() as conn:
        with pytest.raises(ConfigurationError, match=f"--schema {empty} --from {old}"):
            await migrations.apply(conn, migrations.plan_upgrade)


async def test_check_names_a_role_that_cannot_use_the_schema_and_an_empty_search_path(
    schemas: Schemas,
) -> None:
    name, _ = await created(schemas)
    role = f"engine_probe_{uuid4().hex[:8]}"
    await admin(f"CREATE ROLE {role} LOGIN PASSWORD 'probe'")
    try:
        parts = urlsplit(URL)
        url = urlunsplit(parts._replace(netloc=f"{role}:probe@{parts.hostname}:{parts.port}"))
        pool = await asyncpg.create_pool(
            url, min_size=1, max_size=1, server_settings={"search_path": quoted(name)}
        )
        try:
            with pytest.raises(ConfigurationError) as raised:
                await PostgresStorage(pool).check()
        finally:
            await pool.close()
        assert str(raised.value) == (
            f"refinery/postgres: {role} cannot use schema {name} — grant it USAGE on the schema "
            "and SELECT, INSERT, UPDATE, DELETE on its tables"
        )
    finally:
        await admin(f"DROP ROLE {role}")
    empty = await schemas.pool("")
    with pytest.raises(
        ConfigurationError, match=r"^refinery/postgres: no schema on search_path '\"\"' exists"
    ):
        await PostgresStorage(empty).check()


def test_the_first_schema_of_a_search_path_is_read_as_postgres_reads_it() -> None:
    assert migrations._first('"$user", public', "engine", None) == "public"
    assert migrations._first('"$user", public', "engine", "engine") == "engine"
    assert migrations._first('"Odd ""x""", public', "engine", None) == 'Odd "x"'
    assert migrations._first("Engine, public", "engine", None) == "engine"
    assert migrations._first("", "engine", None) is None
    assert migrations._first('""', "engine", None) is None


async def test_check_lists_tables_initialize_made_elsewhere_without_choosing_them(
    schemas: Schemas,
) -> None:
    old, _ = await initialized(schemas)
    empty = await schemas.create()
    with pytest.raises(ConfigurationError) as raised:
        await PostgresStorage(await schemas.pool(empty)).check()
    assert str(raised.value) == (
        f"refinery/postgres: no engine schema in {empty} — run rfnry-engine create --schema "
        f"{empty}; if the tables initialize() made in {old} are this app's, run rfnry-engine "
        f"upgrade --schema {empty} --from that schema instead"
    )


async def test_check_never_sends_engine_into_another_librarys_schema(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = await schemas.create()
    pool = await schemas.pool(other)
    async with pool.acquire() as conn:
        await conn.execute("CREATE TABLE memory_sources (id TEXT)")
    old, _ = await initialized(schemas)
    monkeypatch.setattr(migrations, "_shared", lambda schema: schema == old)
    home = schemas.name()
    monkeypatch.setattr(migrations, "DEFAULT", home)

    async def problem() -> str:
        with pytest.raises(ConfigurationError) as raised:
            await PostgresStorage(pool).check()
        return str(raised.value)

    assert await problem() == (
        f"refinery/postgres: no engine schema in {other} — the engine tables initialize() made "
        f"are in {old} — run rfnry-engine upgrade --schema {home} --from {old} and put it first "
        "on the pool's search_path"
    )
    await admin(f"CREATE SCHEMA {quoted(home)}")
    async with (await schemas.pool(home)).acquire() as conn:
        await migrations.apply(conn, partial(migrations.plan_create, fresh=True))
    assert await problem() == (
        f"refinery/postgres: no engine schema in {other} — engine's tables are in {home} — "
        "put it first on the pool's search_path"
    )


async def test_create_refuses_while_tables_initialize_made_sit_in_a_shared_schema(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, _ = await initialized(schemas)
    monkeypatch.setattr(migrations, "_shared", lambda schema: schema == old)
    name = await schemas.create()
    async with (await schemas.pool(name)).acquire() as conn:
        with pytest.raises(
            ConfigurationError,
            match=f"^refinery/migrations: the engine tables initialize\\(\\) made are in {old} — "
            f"run rfnry-engine upgrade --schema {name} --from {old} to keep their runs, or "
            "create with --fresh to start without them$",
        ):
            await migrations.apply(conn, migrations.plan_create)
        await migrations.apply(conn, partial(migrations.plan_create, fresh=True))
    await PostgresStorage(await schemas.pool(name)).check()


async def test_check_names_the_grants_a_role_lacks(schemas: Schemas) -> None:
    name, _ = await created(schemas)
    role = f"engine_probe_{uuid4().hex[:8]}"
    await admin(f"CREATE ROLE {role} LOGIN PASSWORD 'probe'")
    try:
        await admin(f"GRANT USAGE ON SCHEMA {quoted(name)} TO {role}")
        await admin(f"GRANT SELECT ON ALL TABLES IN SCHEMA {quoted(name)} TO {role}")
        parts = urlsplit(URL)
        url = urlunsplit(parts._replace(netloc=f"{role}:probe@{parts.hostname}:{parts.port}"))
        pool = await asyncpg.create_pool(
            url, min_size=1, max_size=1, server_settings={"search_path": quoted(name)}
        )
        try:
            with pytest.raises(ConfigurationError) as raised:
                await PostgresStorage(pool).check()
            assert str(raised.value) == (
                f"refinery/postgres: {role} lacks INSERT, UPDATE, DELETE on engine_labels and "
                "engine_runs — grant it SELECT, INSERT, UPDATE, DELETE on them and SELECT on "
                "engine_migrations"
            )
            await admin(
                f"GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {quoted(name)} TO {role}"
            )
            await PostgresStorage(pool).check()
        finally:
            await pool.close()
    finally:
        await admin(f"DROP OWNED BY {role}")
        await admin(f"DROP ROLE {role}")


ROWS = Path(__file__).parents[1] / "fixtures" / "rows"
AGENTS = Path(__file__).parents[1] / "fixtures" / "agents"
RELEASES = ["0.1.0", "0.1.1", "0.1.2", "0.1.3", "0.1.4", "0.1.5", "0.1.6", "0.1.7", "0.1.8"]
RUN_COLUMNS = (
    "scope, run_id, parent_run_id, agent, kind, task, at, recorded, duration_ms, model, body"
)


async def written_as_released(conn: Any, row: dict[str, Any], *, tagged: bool) -> None:
    columns, values = RUN_COLUMNS, "$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::text::json"
    args = [
        row["scope"],
        row["run_id"],
        row["parent_run_id"],
        row["agent"],
        row["kind"],
        row["task"],
        datetime.fromisoformat(row["at"]),
        datetime.fromisoformat(row["recorded"]),
        row["duration_ms"],
        row["model"],
        json.dumps(row["body"]),
    ]
    if tagged:
        columns, values = columns + ", tags", values + ", $12::text::jsonb"
        args.append(stored_tags(row["body"]))
    await conn.execute(f"INSERT INTO engine_runs ({columns}) VALUES ({values})", *args)


async def released(schemas: Schemas, version: str) -> tuple[Any, dict[str, Any]]:
    held = json.loads((ROWS / f"{version}.json").read_text(encoding="utf-8"))
    if version >= "0.1.5":
        schema = 3 if version >= "0.1.8" else 2
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:schema])
            patched.setattr(migrations, "LATEST", schema)
            _, pool = await created(schemas)
    else:
        _, pool = await initialized(schemas, WITH_TAGS if version == "0.1.4" else BEFORE_TAGS)
    tagged = version >= "0.1.4"
    async with pool.acquire() as conn:
        for row in held["engine_runs"]:
            await written_as_released(conn, row, tagged=tagged)
        for label in held["engine_labels"]:
            await conn.execute(
                "INSERT INTO engine_labels (scope, label_id, run_id, source, at, body) "
                "VALUES ($1, $2, $3, $4, $5, $6::text::json)",
                label["scope"],
                label["label_id"],
                label["run_id"],
                label["source"],
                datetime.fromisoformat(label["at"]),
                json.dumps(label["body"]),
            )
    return pool, held


def recorded_by(held: dict[str, Any], agent: str) -> dict[str, Any]:
    found: dict[str, Any] = next(row for row in held["engine_runs"] if row["agent"] == agent)
    return found


@pytest.mark.parametrize("version", RELEASES)
async def test_upgrade_brings_every_run_and_label_a_release_recorded_to_this_version_s_shape(
    schemas: Schemas, version: str
) -> None:
    pool, held = await released(schemas, version)
    async with pool.acquire() as conn:
        await migrations.apply(conn, migrations.plan_upgrade)
    storage = PostgresStorage(pool)
    refinery = Refinery(storage)
    await storage.check()
    rows = await storage.runs("acme", limit=None)
    assert sorted(row.run_id for row in rows) == sorted(
        row["run_id"] for row in held["engine_runs"]
    )
    for row in rows:
        assert row.body == flatten(read_back(row, {}), row.recorded)[0].body
    [label] = await storage.labels("acme")
    assert label.body == label_of(label).model_dump(mode="json")
    desk, helper, triage = (recorded_by(held, agent) for agent in ("desk", "helper", "triage"))
    lead = await refinery.run("acme", desk["run_id"])
    assert isinstance(lead, Generation)
    body = desk["body"]
    sent = body["messages"] if "messages" in body else body["conversation"][: body["sent"]]
    assert lead.messages == [Message.model_validate(message) for message in sent]
    assert lead.conversation == [Message.model_validate(m) for m in body["conversation"]]
    assert [step.by for step in lead.steps] == [None, None]
    [child] = lead.children
    tagged = None if child.call is None else dict(child.call.tags)
    assert tagged == (None if "call" not in helper["body"] else {"ticket": "T-1"})
    assert isinstance(await refinery.run("acme", triage["run_id"]), Decision)
    exported = await refinery.export("acme", MemoryBucket(), holdout_percent=0)
    assert exported.rows == {"generation": 2, "preference": 1, "decision": 1}
    teaching = Reply(
        tool_calls=[
            ToolCall(
                id="t1",
                name="Output",
                args={"answers": {"urgent": True}, "rationale": "An outage is urgent."},
            )
        ],
        stop_reason="tool_use",
    )
    judging = Fixed(
        {
            "grounded": CheckAnswer(probability=0.9),
            "done": ScoreAnswer(score=2, probabilities=[0.0, 0.1, 0.9]),
        }
    )
    async with Engine(AGENTS, models={"strong": Scripted(teaching), "judge": judging}) as engine:
        labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
    assert (labeled.judged, labeled.taught, labeled.errors) == (2, 1, {})


async def test_upgrade_to_schema_4_drops_the_task_column_and_every_body_s_task(
    schemas: Schemas,
) -> None:
    pool, held = await released(schemas, "0.1.8")
    desk = recorded_by(held, "desk")["run_id"]
    storage = PostgresStorage(pool)
    async with pool.acquire() as conn:
        name = await conn.fetchval("SELECT current_schema()")
        assert await conn.fetchval("SELECT task FROM engine_runs WHERE run_id = $1", desk) == (
            "triage"
        )
    with pytest.raises(
        ConfigurationError,
        match=f"^refinery/postgres: {name} is at version 3, this rfnry-engine needs 4 — "
        f"run rfnry-engine upgrade --schema {name}$",
    ):
        await storage.check()
    async with pool.acquire() as conn:
        await migrations.apply(conn, migrations.plan_upgrade)
        columns = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'engine_runs'",
            name,
        )
        version = await conn.fetchval("SELECT max(version) FROM engine_migrations")
        bodies = [
            json.loads(row["body"])
            for row in await conn.fetch("SELECT body::text AS body FROM engine_runs")
        ]
    assert "task" not in {row["column_name"] for row in columns}
    assert version == 4
    assert len(bodies) == len(held["engine_runs"]) and all("task" not in b for b in bodies)
    lead = await Refinery(storage).run("acme", desk)
    assert isinstance(lead, Generation) and lead.input == "Where is A-1?"


async def test_a_body_already_in_this_version_s_shape_is_left_as_it_is(schemas: Schemas) -> None:
    pool, held = await released(schemas, "0.1.8")
    [current] = flatten(generation_record(agent="desk"), AT)
    written = "SELECT run_id, xmin::text AS x FROM engine_runs ORDER BY run_id"
    async with pool.acquire() as conn:
        await written_as_released(
            conn, {**current.model_dump(mode="json"), "task": None}, tagged=True
        )
        before = {row["run_id"]: row["x"] for row in await conn.fetch(written)}
        await migrations.apply(conn, migrations.plan_upgrade)
        after = {row["run_id"]: row["x"] for row in await conn.fetch(written)}
    unchanged = {run_id for run_id in before if before[run_id] == after[run_id]}
    assert unchanged == {recorded_by(held, "triage")["run_id"], current.run_id}


@pytest.mark.parametrize("table", ["engine_runs", "engine_labels"])
async def test_a_body_no_release_recorded_stops_the_upgrade_naming_it(
    schemas: Schemas, table: str
) -> None:
    name, pool = await initialized(schemas, BEFORE_TAGS)
    async with pool.acquire() as conn:
        if table == "engine_runs":
            await conn.execute(
                f"INSERT INTO engine_runs ({RUN_COLUMNS}) VALUES "
                "('acme', 'r9', NULL, 'lead', 'generation', NULL, $1, $1, 1, 'm', '{\"reply\": \"x\"}')",
                AT,
            )
        else:
            await conn.execute(
                "INSERT INTO engine_labels (scope, label_id, run_id, source, at, body) "
                "VALUES ('acme', 'l9', 'r1', 'human', $1, '{\"stance\": \"approval\"}')",
                AT,
            )
        with pytest.raises(ConfigurationError) as caught:
            await migrations.apply(conn, migrations.plan_upgrade)
        assert await tables(pool, name) == TABLES
    named = "run 'r9'" if table == "engine_runs" else "label 'l9'"
    assert str(caught.value).startswith(
        f"refinery/migrations: {named} of scope 'acme' holds a body no release of rfnry-engine "
        "recorded ("
    )
    assert str(caught.value).endswith("; delete it or restore its body, then upgrade again")
