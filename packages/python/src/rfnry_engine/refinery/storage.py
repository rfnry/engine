from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class RunRow(BaseModel):
    scope: str
    run_id: str
    parent_run_id: str | None
    agent: str
    kind: str
    task: str | None
    at: datetime
    recorded: datetime
    duration_ms: int
    model: str | None
    body: dict[str, Any]


class LabelRow(BaseModel):
    scope: str
    label_id: str
    run_id: str
    source: str
    at: datetime
    body: dict[str, Any]


@runtime_checkable
class Storage(Protocol):
    async def initialize(self) -> None: ...

    async def put_run(self, scope: str, row: RunRow) -> None: ...

    async def get_run(self, scope: str, run_id: str) -> RunRow | None: ...

    async def runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        kind: str | None = None,
        since: datetime | None = None,
        unlabeled_by: str | None = None,
        limit: int | None = 1000,
    ) -> list[RunRow]: ...

    async def put_label(self, scope: str, row: LabelRow) -> LabelRow: ...

    async def labels(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> list[LabelRow]: ...

    async def run_tree(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[RunRow]: ...

    async def delete_runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> int: ...

    async def delete_scope(self, scope: str) -> int: ...

    async def count_scope(self, scope: str) -> int: ...
