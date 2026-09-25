from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.conflicts import find_conflicts
from rfnry_engine.refinery.export import (
    Bucket,
    decision_rows,
    export_key,
    generation_row,
    preference_rows,
)
from rfnry_engine.refinery.holdout import HOLDOUT_PERCENT, is_holdout
from rfnry_engine.refinery.judge import (
    TeacherAnswers,
    judge_state,
    teacher_answers,
    teacher_prompt,
)
from rfnry_engine.refinery.labels import Conflict, Exported, Label, Labeled, Outcome, Review
from rfnry_engine.refinery.storage import LabelRow, RunRow, Storage


def flatten(run: Run) -> list[RunRow]:
    body = run.model_dump(mode="json")
    children: list[Run] = []
    if isinstance(run, Generation):
        body["children"] = []
        children = list(run.children)
    elif isinstance(run, Decision):
        body["routed_to"] = None
        children = [run.routed_to] if run.routed_to is not None else []
    row = RunRow(
        scope=run.scope,
        run_id=run.run_id,
        parent_run_id=run.parent_run_id,
        agent=run.agent,
        kind=run.kind,
        task=run.task if isinstance(run, Generation) else None,
        at=run.at,
        duration_ms=run.duration_ms,
        model=run.model,
        body=body,
    )
    rows = [row]
    for child in children:
        rows.extend(flatten(child))
    return rows


def label_of(row: LabelRow) -> Label:
    return Label.model_validate(row.body)


class Refinery:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def initialize(self) -> None:
        await self.storage.initialize()

    async def record(self, run: Run) -> None:
        for row in flatten(run):
            await self.storage.put_run(row.scope, row)

    async def _run_row(self, scope: str, run_id: str) -> RunRow:
        row = await self.storage.get_run(scope, run_id)
        if row is None:
            raise NotFoundError(f"refinery: run {run_id!r} was not found")
        return row

    async def _put(self, label: Label) -> Label:
        await self.storage.put_label(
            label.scope,
            LabelRow(
                scope=label.scope,
                label_id=label.label_id,
                run_id=label.run_id,
                source=label.source,
                at=label.at,
                body=label.model_dump(mode="json"),
            ),
        )
        return label

    async def review(self, scope: str, run_id: str, review: Review) -> Label:
        row = await self._run_row(scope, run_id)
        if row.kind == "generation" and isinstance(review.correction, dict):
            raise InputError("refinery: a generation run takes a text correction")
        if row.kind == "decision" and isinstance(review.correction, str):
            raise InputError("refinery: a decision run takes a correction per question id")
        return await self._put(
            Label(
                label_id=uuid4().hex,
                run_id=run_id,
                scope=scope,
                at=datetime.now(UTC),
                source="human",
                review=review,
            )
        )

    async def outcome(self, scope: str, run_id: str, outcome: Outcome) -> Label:
        await self._run_row(scope, run_id)
        return await self._put(
            Label(
                label_id=uuid4().hex,
                run_id=run_id,
                scope=scope,
                at=datetime.now(UTC),
                source="world",
                outcome=outcome,
            )
        )

    async def conflicts(self, scope: str) -> list[Conflict]:
        rows = await self.storage.labels(scope, source="human")
        return find_conflicts([label_of(row) for row in rows])

    async def export(
        self,
        scope: str,
        bucket: Bucket,
        *,
        since: datetime | None = None,
        holdout_percent: int = HOLDOUT_PERCENT,
    ) -> Exported:
        runs = await self.storage.runs(scope, since=since, limit=None)
        if since is not None:
            fresh = {run.run_id for run in runs}
            relabeled = {row.run_id for row in await self.storage.labels(scope, since=since)} - fresh
            if relabeled:
                runs += await self.storage.runs(scope, run_ids=sorted(relabeled), limit=None)
        if not runs:
            return Exported(keys=[], rows={}, watermark=since)
        labels: dict[str, list[Label]] = {}
        stamps = [run.at for run in runs]
        for row in await self.storage.labels(scope, run_ids=[r.run_id for r in runs]):
            labels.setdefault(row.run_id, []).append(label_of(row))
            stamps.append(row.at)
        groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        for run in runs:
            holdout = is_holdout(run.run_id, holdout_percent=holdout_percent)
            own = labels.get(run.run_id, [])
            if run.kind == "generation":
                groups.setdefault((run.agent, "generation"), []).append(
                    generation_row(run, own, holdout=holdout)
                )
                groups.setdefault((run.agent, "preference"), []).extend(
                    preference_rows(run, own, holdout=holdout)
                )
            else:
                groups.setdefault((run.agent, "decision"), []).extend(
                    decision_rows(run, own, holdout=holdout)
                )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        keys: list[str] = []
        counts: dict[str, int] = {}
        for (agent, dataset), rows in sorted(groups.items()):
            if not rows:
                continue
            key = export_key(scope, agent, dataset, stamp)
            await bucket.put(key, ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8"))
            keys.append(key)
            counts[dataset] = counts.get(dataset, 0) + len(rows)
        return Exported(keys=keys, rows=counts, watermark=max(stamps))

    async def label(
        self, scope: str, engine: Engine, *, judge: str | None = None, teacher: str | None = None, limit: int = 500
    ) -> Labeled:
        labelers = {name for name in (judge, teacher) if name is not None}
        result = Labeled()
        if judge is not None:
            for row in await self.storage.runs(scope, kind="generation", unlabeled_by="judge", limit=limit):
                if row.agent in labelers:
                    continue
                try:
                    run = await engine.run(judge, scope, judge_state(row), record=False)
                    if not isinstance(run, Decision):
                        raise ConfigurationError(f"refinery: judge {judge!r} must be a decision agent")
                    await self._put(Label(label_id=uuid4().hex, run_id=row.run_id, scope=scope, at=datetime.now(UTC), source="judge", answers=run.answers))
                    result.judged += 1
                except Exception as exc:
                    result.errors[row.run_id] = f"{type(exc).__name__}: {exc}"
        if teacher is not None:
            for row in await self.storage.runs(scope, kind="decision", unlabeled_by="teacher", limit=limit):
                if row.agent in labelers:
                    continue
                try:
                    run = await engine.run(teacher, scope, teacher_prompt(row), expect=TeacherAnswers, record=False)
                    if not isinstance(run, Generation):
                        raise ConfigurationError(f"refinery: teacher {teacher!r} must be a generation agent")
                    questions = Decision.model_validate(row.body).questions
                    answers = teacher_answers(questions, TeacherAnswers.model_validate_json(run.reply))
                    await self._put(Label(label_id=uuid4().hex, run_id=row.run_id, scope=scope, at=datetime.now(UTC), source="teacher", answers=answers))
                    result.taught += 1
                except Exception as exc:
                    result.errors[row.run_id] = f"{type(exc).__name__}: {exc}"
        return result

    async def delete_scope(self, scope: str) -> int:
        return await self.storage.delete_scope(scope)

    async def verify_absent(self, scope: str) -> bool:
        return await self.storage.count_scope(scope) == 0
