from __future__ import annotations

from datetime import datetime

from rfnry_engine.errors import StorageError
from rfnry_engine.refinery.storage import LabelRow, RunRow


class InMemoryStorage:
    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], RunRow] = {}
        self._labels: dict[tuple[str, str], LabelRow] = {}

    async def initialize(self) -> None:
        return None

    async def put_run(self, scope: str, row: RunRow) -> None:
        if row.scope != scope:
            raise StorageError("inmemory: row scope does not match the call scope")
        self._runs[(scope, row.run_id)] = row

    async def get_run(self, scope: str, run_id: str) -> RunRow | None:
        return self._runs.get((scope, run_id))

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
        labeled = {
            label.run_id
            for (s, _), label in self._labels.items()
            if s == scope and unlabeled_by is not None and label.source == unlabeled_by
        }
        wanted = set(run_ids) if run_ids is not None else None
        rows = [
            row
            for (s, _), row in self._runs.items()
            if s == scope
            and (wanted is None or row.run_id in wanted)
            and (kind is None or row.kind == kind)
            and (since is None or row.at > since)
            and row.run_id not in labeled
        ]
        rows.sort(key=lambda r: (r.at, r.run_id), reverse=True)
        return rows[:limit]

    async def put_label(self, scope: str, row: LabelRow) -> None:
        if row.scope != scope:
            raise StorageError("inmemory: row scope does not match the call scope")
        self._labels[(scope, row.label_id)] = row

    async def labels(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> list[LabelRow]:
        wanted = set(run_ids) if run_ids is not None else None
        rows = [
            row
            for (s, _), row in self._labels.items()
            if s == scope
            and (wanted is None or row.run_id in wanted)
            and (source is None or row.source == source)
            and (since is None or row.at > since)
        ]
        rows.sort(key=lambda r: (r.at, r.label_id))
        return rows

    async def delete_scope(self, scope: str) -> int:
        run_keys = [key for key in self._runs if key[0] == scope]
        label_keys = [key for key in self._labels if key[0] == scope]
        for key in run_keys:
            del self._runs[key]
        for key in label_keys:
            del self._labels[key]
        return len(run_keys) + len(label_keys)

    async def count_scope(self, scope: str) -> int:
        runs = sum(1 for key in self._runs if key[0] == scope)
        labels = sum(1 for key in self._labels if key[0] == scope)
        return runs + labels
