from __future__ import annotations

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
        at TIMESTAMPTZ NOT NULL, duration_ms INTEGER NOT NULL, model TEXT NULL,
        body JSON NOT NULL,
        PRIMARY KEY (scope, run_id))
    """,
    "CREATE INDEX IF NOT EXISTS ix_engine_runs_scope_at ON engine_runs (scope, at)",
    """
    CREATE TABLE IF NOT EXISTS engine_labels (
        scope TEXT NOT NULL, label_id TEXT NOT NULL, run_id TEXT NOT NULL,
        source TEXT NOT NULL, at TIMESTAMPTZ NOT NULL, body JSON NOT NULL,
        PRIMARY KEY (scope, label_id))
    """,
    "CREATE INDEX IF NOT EXISTS ix_engine_labels_scope_run ON engine_labels (scope, run_id)",
]

_RUN_COLUMNS = "scope, run_id, parent_run_id, agent, kind, task, at, duration_ms, model, body"
_LABEL_COLUMNS = "scope, label_id, run_id, source, at, body"


def _json(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


def _run(record: Any) -> RunRow:
    return RunRow(
        scope=record["scope"], run_id=record["run_id"], parent_run_id=record["parent_run_id"], agent=record["agent"],
        kind=record["kind"], task=record["task"], at=record["at"], duration_ms=record["duration_ms"],
        model=record["model"], body=_json(record["body"]),
    )


def _label(record: Any) -> LabelRow:
    return LabelRow(
        scope=record["scope"], label_id=record["label_id"], run_id=record["run_id"], source=record["source"],
        at=record["at"], body=_json(record["body"]),
    )


def _count(status: str) -> int:
    return int(status.rsplit(" ", 1)[-1])


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
            for ddl in _DDL:
                await conn.execute(ddl)

    async def put_run(self, scope: str, row: RunRow) -> None:
        if row.scope != scope:
            raise StorageError("postgres: row scope does not match the call scope")
        async with self._guard("put_run"), self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO engine_runs ({_RUN_COLUMNS}) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::json) "
                "ON CONFLICT (scope, run_id) DO UPDATE SET parent_run_id = EXCLUDED.parent_run_id, agent = EXCLUDED.agent, "
                "kind = EXCLUDED.kind, task = EXCLUDED.task, at = EXCLUDED.at, duration_ms = EXCLUDED.duration_ms, "
                "model = EXCLUDED.model, body = EXCLUDED.body",
                row.scope, row.run_id, row.parent_run_id, row.agent, row.kind, row.task, row.at,
                row.duration_ms, row.model, json.dumps(row.body),
            )

    async def get_run(self, scope: str, run_id: str) -> RunRow | None:
        async with self._guard("get_run"), self._pool.acquire() as conn:
            record = await conn.fetchrow(
                f"SELECT {_RUN_COLUMNS} FROM engine_runs WHERE scope = $1 AND run_id = $2", scope, run_id
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
            conditions.append(f"r.at > ${len(params)}")
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

    async def put_label(self, scope: str, row: LabelRow) -> None:
        if row.scope != scope:
            raise StorageError("postgres: row scope does not match the call scope")
        async with self._guard("put_label"), self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO engine_labels ({_LABEL_COLUMNS}) VALUES ($1, $2, $3, $4, $5, $6::json) "
                "ON CONFLICT (scope, label_id) DO UPDATE SET run_id = EXCLUDED.run_id, source = EXCLUDED.source, "
                "at = EXCLUDED.at, body = EXCLUDED.body",
                row.scope, row.label_id, row.run_id, row.source, row.at, json.dumps(row.body),
            )

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
        sql = f"SELECT {_LABEL_COLUMNS} FROM engine_labels WHERE " + " AND ".join(conditions) + ' ORDER BY at ASC, label_id COLLATE "C" ASC'
        async with self._guard("labels"), self._pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        return [_label(record) for record in records]

    async def delete_scope(self, scope: str) -> int:
        async with self._guard("delete_scope"), self._pool.acquire() as conn, conn.transaction():
            labels = await conn.execute("DELETE FROM engine_labels WHERE scope = $1", scope)
            runs = await conn.execute("DELETE FROM engine_runs WHERE scope = $1", scope)
        return _count(labels) + _count(runs)

    async def count_scope(self, scope: str) -> int:
        async with self._guard("count_scope"), self._pool.acquire() as conn:
            runs = await conn.fetchval("SELECT count(*) FROM engine_runs WHERE scope = $1", scope)
            labels = await conn.fetchval("SELECT count(*) FROM engine_labels WHERE scope = $1", scope)
        return int(runs) + int(labels)
