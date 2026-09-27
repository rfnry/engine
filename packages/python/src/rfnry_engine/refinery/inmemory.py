from __future__ import annotations

from datetime import datetime

from rfnry_engine.errors import StorageError
from rfnry_engine.refinery.storage import LabelRow, RunRow


def _tagged(row: RunRow, tags: dict[str, str]) -> bool:
    held = row.body.get("tags")
    return isinstance(held, dict) and all(
        key in held and held[key] == value for key, value in tags.items()
    )


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
            and (since is None or row.recorded > since)
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

    def _tree(
        self, scope: str, run_ids: list[str] | None, tags: dict[str, str] | None
    ) -> list[RunRow]:
        rows = [row for (s, _), row in self._runs.items() if s == scope]
        wanted = set(run_ids) if run_ids is not None else None
        found = {
            row.run_id
            for row in rows
            if (wanted is None or row.run_id in wanted or row.parent_run_id in wanted)
            and (tags is None or _tagged(row, tags))
        }
        grown = found
        while grown:
            grown = {row.run_id for row in rows if row.parent_run_id in grown} - found
            found |= grown
        return sorted((row for row in rows if row.run_id in found), key=lambda r: (r.at, r.run_id))

    async def run_tree(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[RunRow]:
        return self._tree(scope, run_ids, tags)

    async def delete_runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> int:
        found = {row.run_id for row in self._tree(scope, run_ids, tags)}
        labeled = found | set(run_ids or [])
        label_keys = [
            key
            for key, label in self._labels.items()
            if key[0] == scope and label.run_id in labeled
        ]
        for key in label_keys:
            del self._labels[key]
        for run_id in found:
            del self._runs[(scope, run_id)]
        return len(label_keys) + len(found)

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
