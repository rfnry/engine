import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from rfnry_engine.errors import StorageError
from rfnry_engine.refinery.storage import LabelRow, RunRow

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS engine_runs (
        scope TEXT NOT NULL, run_id TEXT NOT NULL, parent_run_id TEXT NULL,
        agent TEXT NOT NULL, kind TEXT NOT NULL, task TEXT NULL,
        at TIMESTAMPTZ NOT NULL, recorded TIMESTAMPTZ NOT NULL,
        duration_ms INTEGER NOT NULL, model TEXT NULL, body JSON NOT NULL, tags JSONB NOT NULL,
        PRIMARY KEY (scope, run_id))
    """,
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_at ON engine_runs (scope, at)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_recorded ON engine_runs (scope, recorded)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_parent ON engine_runs (scope, parent_run_id)",
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_tags ON engine_runs USING GIN (tags jsonb_path_ops)",
    """
    CREATE TABLE IF NOT EXISTS engine_labels (
        scope TEXT NOT NULL, label_id TEXT NOT NULL, run_id TEXT NOT NULL,
        source TEXT NOT NULL, at TIMESTAMPTZ NOT NULL, body JSON NOT NULL,
        PRIMARY KEY (scope, label_id))
    """,
    "CREATE INDEX IF NOT EXISTS ix_engine_labels_scope_run ON engine_labels (scope, run_id)",
]

_RUN_COLUMNS = (
    "scope, run_id, parent_run_id, agent, kind, task, at, recorded, duration_ms, model, body"
)
_LABEL_COLUMNS = "scope, label_id, run_id, source, at, body"
_TABLES = {
    "engine_runs": [*_RUN_COLUMNS.split(", "), "tags"],
    "engine_labels": _LABEL_COLUMNS.split(", "),
}


def _json(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


def _run(record: Any) -> RunRow:
    return RunRow(
        scope=record["scope"],
        run_id=record["run_id"],
        parent_run_id=record["parent_run_id"],
        agent=record["agent"],
        kind=record["kind"],
        task=record["task"],
        at=record["at"],
        recorded=record["recorded"],
        duration_ms=record["duration_ms"],
        model=record["model"],
        body=_json(record["body"]),
    )


def _label(record: Any) -> LabelRow:
    return LabelRow(
        scope=record["scope"],
        label_id=record["label_id"],
        run_id=record["run_id"],
        source=record["source"],
        at=record["at"],
        body=_json(record["body"]),
    )


def _count(status: str) -> int:
    return int(status.rsplit(" ", 1)[-1])


def _as_jsonb(text: str) -> str:
    return f"replace(replace({text}, '\\ud', '\\u0'), '\\u0000', '\\ufffd')::jsonb"


def _tree(
    scope: str, run_ids: list[str] | None, tags: dict[str, str] | None
) -> tuple[str, list[Any]]:
    conditions = ["scope = $1"]
    params: list[Any] = [scope]
    if run_ids is not None:
        params.append(list(run_ids))
        conditions.append(
            f"(run_id = ANY(${len(params)}::text[]) OR parent_run_id = ANY(${len(params)}::text[]))"
        )
    if tags is not None:
        params.append(json.dumps(tags))
        conditions.append(f"tags @> {_as_jsonb(f'${len(params)}::text')}")
    sql = (
        "WITH RECURSIVE tree(run_id) AS (SELECT run_id FROM engine_runs WHERE "
        + " AND ".join(conditions)
        + " UNION SELECT r.run_id FROM engine_runs r JOIN tree t ON r.parent_run_id = t.run_id "
        "WHERE r.scope = $1) "
    )
    return sql, params


class PostgresStorage:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
        try:
            yield
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"postgres: storage rejected {operation}: {exc}") from exc

    async def initialize(self) -> None:
        async with self._guard("initialize"), self._pool.acquire() as conn:
            found = await conn.fetch(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = ANY($1::text[])",
                list(_TABLES),
            )
            for table, columns in _TABLES.items():
                held = {record["column_name"] for record in found if record["table_name"] == table}
                missing = [column for column in columns if held and column not in held]
                if missing:
                    raise StorageError(
                        f"postgres: {table} was made by an earlier version and lacks "
                        f"{', '.join(missing)}; drop engine_runs and engine_labels, then run "
                        "initialize() again"
                    )
            for ddl in _DDL:
                await conn.execute(ddl)

    async def put_runs(self, scope: str, rows: list[RunRow]) -> None:
        if any(row.scope != scope for row in rows):
            raise StorageError("postgres: row scope does not match the call scope")
        if not rows:
            return
        async with self._guard("put_runs"), self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                f"INSERT INTO engine_runs ({_RUN_COLUMNS}, tags) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::json, "
                f"{_as_jsonb('$12::text')}) "
                "ON CONFLICT (scope, run_id) DO UPDATE SET parent_run_id = EXCLUDED.parent_run_id, agent = EXCLUDED.agent, "
                "kind = EXCLUDED.kind, task = EXCLUDED.task, at = EXCLUDED.at, recorded = EXCLUDED.recorded, "
                "duration_ms = EXCLUDED.duration_ms, model = EXCLUDED.model, body = EXCLUDED.body, tags = EXCLUDED.tags",
                [
                    (
                        row.scope,
                        row.run_id,
                        row.parent_run_id,
                        row.agent,
                        row.kind,
                        row.task,
                        row.at,
                        row.recorded,
                        row.duration_ms,
                        row.model,
                        json.dumps(row.body),
                        json.dumps(row.body.get("tags") or {}),
                    )
                    for row in rows
                ],
            )

    async def get_run(self, scope: str, run_id: str) -> RunRow | None:
        async with self._guard("get_run"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                f"SELECT {_RUN_COLUMNS} FROM engine_runs WHERE scope = $1 AND run_id = $2",
                scope,
                run_id,
            )
        return _run(record) if record is not None else None

    async def runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        kind: str | None = None,
        since: datetime | None = None,
        unlabeled_by: str | None = None,
        limit: int | None = 1000,
    ) -> list[RunRow]:
        conditions = ["r.scope = $1"]
        params: list[Any] = [scope]
        if run_ids is not None:
            params.append(list(run_ids))
            conditions.append(f"r.run_id = ANY(${len(params)}::text[])")
        if kind is not None:
            params.append(kind)
            conditions.append(f"r.kind = ${len(params)}")
        if since is not None:
            params.append(since)
            conditions.append(f"r.recorded > ${len(params)}")
        if unlabeled_by is not None:
            params.append(unlabeled_by)
            conditions.append(
                "NOT EXISTS (SELECT 1 FROM engine_labels l WHERE l.scope = r.scope AND l.run_id = r.run_id "
                f"AND l.source = ${len(params)})"
            )
        sql = (
            f"SELECT {', '.join('r.' + c for c in _RUN_COLUMNS.split(', '))} FROM engine_runs r WHERE "
            + " AND ".join(conditions)
            + ' ORDER BY r.at DESC, r.run_id COLLATE "C" DESC'
        )
        if limit is not None:
            params.append(limit)
            sql += f" LIMIT ${len(params)}"
        async with self._guard("runs"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        return [_run(record) for record in records]

    async def put_label(self, scope: str, row: LabelRow) -> LabelRow:
        if row.scope != scope:
            raise StorageError("postgres: row scope does not match the call scope")
        async with self._guard("put_label"), self._pool.acquire() as conn:
            inserted = await conn.fetchval(
                f"INSERT INTO engine_labels ({_LABEL_COLUMNS}) VALUES ($1, $2, $3, $4, $5, $6::json) "
                "ON CONFLICT (scope, label_id) DO NOTHING RETURNING true",
                row.scope,
                row.label_id,
                row.run_id,
                row.source,
                row.at,
                json.dumps(row.body),
            )
            if inserted:
                return row
            record = await conn.fetchrow(
                f"SELECT {_LABEL_COLUMNS} FROM engine_labels WHERE scope = $1 AND label_id = $2",
                scope,
                row.label_id,
            )
            return _label(record)

    async def labels(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> list[LabelRow]:
        conditions = ["scope = $1"]
        params: list[Any] = [scope]
        if run_ids is not None:
            params.append(list(run_ids))
            conditions.append(f"run_id = ANY(${len(params)}::text[])")
        if source is not None:
            params.append(source)
            conditions.append(f"source = ${len(params)}")
        if since is not None:
            params.append(since)
            conditions.append(f"at > ${len(params)}")
        sql = (
            f"SELECT {_LABEL_COLUMNS} FROM engine_labels WHERE "
            + " AND ".join(conditions)
            + ' ORDER BY at ASC, label_id COLLATE "C" ASC'
        )
        async with self._guard("labels"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        return [_label(record) for record in records]

    async def run_tree(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[RunRow]:
        tree, params = _tree(scope, run_ids, tags)
        sql = (
            tree + f"SELECT {_RUN_COLUMNS} FROM engine_runs WHERE scope = $1 "
            'AND run_id IN (SELECT run_id FROM tree) ORDER BY at ASC, run_id COLLATE "C" ASC'
        )
        async with self._guard("run_tree"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        return [_run(record) for record in records]

    async def delete_runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> int:
        tree, params = _tree(scope, run_ids, tags)
        async with self._guard("delete_runs"), self._pool.acquire() as conn, conn.transaction():
            found = [
                r["run_id"] for r in await conn.fetch(tree + "SELECT run_id FROM tree", *params)
            ]
            labels = await conn.execute(
                "DELETE FROM engine_labels WHERE scope = $1 AND run_id = ANY($2::text[])",
                scope,
                found + list(run_ids or []),
            )
            runs = await conn.execute(
                "DELETE FROM engine_runs WHERE scope = $1 AND run_id = ANY($2::text[])",
                scope,
                found,
            )
        return _count(labels) + _count(runs)

    async def delete_scope(self, scope: str) -> int:
        async with self._guard("delete_scope"), self._pool.acquire() as conn, conn.transaction():
            labels = await conn.execute("DELETE FROM engine_labels WHERE scope = $1", scope)
            runs = await conn.execute("DELETE FROM engine_runs WHERE scope = $1", scope)
        return _count(labels) + _count(runs)

    async def count_scope(self, scope: str) -> int:
        async with self._guard("count_scope"), self._pool.acquire() as conn:
            runs = await conn.fetchval("SELECT count(*) FROM engine_runs WHERE scope = $1", scope)
            labels = await conn.fetchval(
                "SELECT count(*) FROM engine_labels WHERE scope = $1", scope
            )
        return int(runs) + int(labels)
