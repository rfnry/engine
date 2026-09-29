from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from rfnry_engine.engine import Approve, Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.records import Decision, Fingerprint, Generation, Run
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
from rfnry_engine.refinery.labels import (
    Conflict,
    Exported,
    Label,
    Labeled,
    Outcome,
    Replay,
    Review,
)
from rfnry_engine.refinery.storage import LabelRow, RunRow, Storage
from rfnry_engine.tools.tools import ToolsLike
from rfnry_engine.tools.workspace import Workspace

REPORTED = ("human", "world")


def flatten(run: Run, recorded: datetime) -> list[RunRow]:
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
        task=run.task,
        at=run.at,
        recorded=recorded,
        duration_ms=run.duration_ms,
        model=run.model,
        body=body,
    )
    rows = [row]
    for child in children:
        rows.extend(flatten(child, recorded))
    return rows


def children_of(rows: list[RunRow]) -> dict[str, list[RunRow]]:
    under: dict[str, list[RunRow]] = {}
    for row in rows:
        if row.parent_run_id is not None:
            under.setdefault(row.parent_run_id, []).append(row)
    return under


def unflatten(row: RunRow, under: dict[str, list[RunRow]]) -> dict[str, Any]:
    body = dict(row.body)
    named = [step.get("child_run_id") for step in body.get("steps", [])]
    order = {run_id: index for index, run_id in enumerate(named) if run_id is not None}
    children = sorted(
        under.get(row.run_id, []),
        key=lambda child: order.get(child.run_id, len(order)),
    )
    nested = [unflatten(child, under) for child in children]
    if row.kind == "generation":
        body["children"] = nested
    else:
        body["routed_to"] = nested[0] if nested else None
    return body


def read_back(row: RunRow, under: dict[str, list[RunRow]]) -> Run:
    body = unflatten(row, under)
    if row.kind == "generation":
        return Generation.model_validate(body)
    return Decision.model_validate(body)


def printed(row: RunRow) -> Fingerprint | None:
    body = row.body.get("fingerprint")
    return None if body is None else Fingerprint.model_validate(body)


def label_of(row: LabelRow) -> Label:
    return Label.model_validate(row.body)


def check_selector(verb: str, run_ids: list[str] | None, tags: dict[str, str] | None) -> None:
    if run_ids is not None and tags is not None:
        raise InputError(f"refinery: {verb} takes run_ids or tags, not both")
    if run_ids is not None and not run_ids:
        raise InputError("refinery: run_ids must name at least one run")
    if tags is not None and not tags:
        raise InputError("refinery: tags must hold at least one tag")


class Refinery:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def initialize(self) -> None:
        await self.storage.initialize()

    async def record(self, run: Run) -> None:
        if not run.scope:
            raise InputError("refinery: scope is required")
        for row in flatten(run, datetime.now(UTC)):
            await self.storage.put_run(row.scope, row)

    async def run(self, scope: str, run_id: str) -> Run:
        if not scope:
            raise InputError("refinery: scope is required")
        rows = await self.storage.run_tree(scope, run_ids=[run_id])
        root = next((row for row in rows if row.run_id == run_id), None)
        if root is None:
            raise NotFoundError(f"refinery: run {run_id!r} was not found")
        return read_back(root, children_of(rows))

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
        if not scope:
            raise InputError("refinery: scope is required")
        row = await self._run_row(scope, run_id)
        if row.kind == "generation" and isinstance(review.correction, dict):
            raise InputError("refinery: a generation run takes a text correction")
        if row.kind == "decision" and isinstance(review.correction, str):
            raise InputError("refinery: a decision run takes a correction per question id")
        if isinstance(review.correction, dict):
            unknown = sorted(set(review.correction) - set(row.body["questions"]))
            if unknown:
                raise InputError(
                    f"refinery: run {run_id!r} asked no question {', '.join(map(repr, unknown))}"
                )
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
        if not scope:
            raise InputError("refinery: scope is required")
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
        if not scope:
            raise InputError("refinery: scope is required")
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
        if not scope:
            raise InputError("refinery: scope is required")
        if not 0 <= holdout_percent <= 100:
            raise InputError(f"refinery: holdout_percent must be 0-100, got {holdout_percent}")
        runs = await self.storage.runs(scope, since=since, limit=None)
        if since is not None:
            fresh = {run.run_id for run in runs}
            relabeled = {
                row.run_id for row in await self.storage.labels(scope, since=since)
            } - fresh
            if relabeled:
                runs += await self.storage.runs(scope, run_ids=sorted(relabeled), limit=None)
        if not runs:
            return Exported(keys=[], rows={}, watermark=since)
        labels: dict[str, list[Label]] = {}
        stamps = [run.recorded for run in runs]
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
            await bucket.put(
                key,
                ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8"),
            )
            keys.append(key)
            counts[dataset] = counts.get(dataset, 0) + len(rows)
        return Exported(keys=keys, rows=counts, watermark=max(stamps))

    async def label(
        self,
        scope: str,
        engine: Engine,
        *,
        judge: str | None = None,
        teacher: str | None = None,
        limit: int = 500,
    ) -> Labeled:
        if not scope:
            raise InputError("refinery: scope is required")
        agents = engine.agents
        for role, name, kind in (("judge", judge, "decision"), ("teacher", teacher, "generation")):
            if name is None:
                continue
            if name not in agents:
                raise NotFoundError(f"refinery: {role} {name!r} was not found")
            if agents[name].kind != kind:
                raise ConfigurationError(f"refinery: {role} {name!r} must be a {kind} agent")
        labelers = {name for name in (judge, teacher) if name is not None}
        result = Labeled()
        if judge is not None:
            for row in await self.storage.runs(
                scope, kind="generation", unlabeled_by="judge", limit=limit
            ):
                if row.agent in labelers:
                    continue
                try:
                    run = await engine.run(judge, scope, judge_state(row), record=False)
                    if not isinstance(run, Decision):
                        raise ConfigurationError(
                            f"refinery: judge {judge!r} must be a decision agent"
                        )
                    await self._put(
                        Label(
                            label_id=uuid4().hex,
                            run_id=row.run_id,
                            scope=scope,
                            at=datetime.now(UTC),
                            source="judge",
                            answers=run.answers,
                        )
                    )
                    result.judged += 1
                except Exception as exc:
                    result.errors[row.run_id] = f"{type(exc).__name__}: {exc}"
        if teacher is not None:
            for row in await self.storage.runs(
                scope, kind="decision", unlabeled_by="teacher", limit=limit
            ):
                if row.agent in labelers:
                    continue
                try:
                    run = await engine.run(
                        teacher, scope, teacher_prompt(row), expect=TeacherAnswers, record=False
                    )
                    if not isinstance(run, Generation):
                        raise ConfigurationError(
                            f"refinery: teacher {teacher!r} must be a generation agent"
                        )
                    questions = Decision.model_validate(row.body).questions
                    answers = teacher_answers(
                        questions, TeacherAnswers.model_validate_json(run.reply)
                    )
                    await self._put(
                        Label(
                            label_id=uuid4().hex,
                            run_id=row.run_id,
                            scope=scope,
                            at=datetime.now(UTC),
                            source="teacher",
                            answers=answers,
                        )
                    )
                    result.taught += 1
                except Exception as exc:
                    result.errors[row.run_id] = f"{type(exc).__name__}: {exc}"
        return result

    def replay(
        self,
        scope: str,
        candidate: Engine,
        *,
        agent: str,
        holdout: bool = False,
        labeled: bool = False,
        tags: dict[str, str] | None = None,
        fingerprint: Fingerprint | None = None,
        limit: int | None = None,
        holdout_percent: int = HOLDOUT_PERCENT,
        expect: type[BaseModel] | None = None,
        approve: Approve | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        concurrency: int = 1,
    ) -> AsyncGenerator[Replay, None]:
        if not scope:
            raise InputError("refinery: scope is required")
        if not 0 <= holdout_percent <= 100:
            raise InputError(f"refinery: holdout_percent must be 0-100, got {holdout_percent}")
        if tags is not None:
            check_selector("replay", None, tags)
        if limit is not None and limit < 1:
            raise InputError(f"refinery: limit must be at least 1, got {limit}")
        if concurrency < 1:
            raise InputError(f"refinery: concurrency must be at least 1, got {concurrency}")
        if agent not in candidate.agents:
            raise NotFoundError(f"refinery: the candidate has no agent {agent!r}")

        async def rerun(recorded: Run) -> Run:
            asked = (
                {qid: recorded.questions[qid] for qid in recorded.runtime}
                if isinstance(recorded, Decision) and recorded.runtime
                else None
            )
            run = recorded
            while isinstance(run, Decision) and run.routed_to is not None:
                run = run.routed_to
            if isinstance(run, Decision):
                return await candidate.run(
                    agent,
                    scope,
                    run.state,
                    task=run.task,
                    questions=asked,
                    expect=expect,
                    approve=approve,
                    tools=tools,
                    workspace=workspace,
                    record=False,
                )
            sent = run.messages[-1].content if run.messages else None
            return await candidate.run(
                agent,
                scope,
                sent if isinstance(sent, list) else run.input,
                task=run.task,
                questions=asked,
                history=run.messages[:-1],
                context=run.context,
                expect=expect,
                approve=approve,
                tools=tools,
                workspace=workspace,
                record=False,
            )

        async def pair(recorded: Run, labels: list[Label]) -> Replay:
            try:
                run = await rerun(recorded)
            except Exception as exc:
                return Replay(
                    recorded=recorded, labels=labels, error=f"{type(exc).__name__}: {exc}"
                )
            return Replay(recorded=recorded, labels=labels, candidate=run)

        async def pairs() -> AsyncGenerator[Replay, None]:
            chosen = await self._replayable(
                scope,
                agent,
                holdout=holdout,
                holdout_percent=holdout_percent,
                labeled=labeled,
                tags=tags,
                fingerprint=fingerprint,
                limit=limit,
            )
            running: deque[asyncio.Task[Replay]] = deque()

            async def first() -> Replay:
                done = await running[0]
                running.popleft()
                return done

            try:
                for recorded, labels in chosen:
                    running.append(asyncio.ensure_future(pair(recorded, labels)))
                    if len(running) == concurrency:
                        yield await first()
                while running:
                    yield await first()
            finally:
                for task in running:
                    task.cancel()
                await asyncio.gather(*running, return_exceptions=True)

        return pairs()

    async def _replayable(
        self,
        scope: str,
        agent: str,
        *,
        holdout: bool,
        holdout_percent: int,
        labeled: bool,
        tags: dict[str, str] | None,
        fingerprint: Fingerprint | None,
        limit: int | None,
    ) -> list[tuple[Run, list[Label]]]:
        rows = await self.storage.run_tree(scope, tags=tags)
        tops = [
            row
            for row in rows
            if row.parent_run_id is None
            and row.agent == agent
            and (not holdout or is_holdout(row.run_id, holdout_percent=holdout_percent))
            and (fingerprint is None or printed(row) == fingerprint)
        ]
        labels: dict[str, list[Label]] = {}
        if tops:
            for row in await self.storage.labels(scope, run_ids=[top.run_id for top in tops]):
                labels.setdefault(row.run_id, []).append(label_of(row))
        if labeled:
            tops = [
                top
                for top in tops
                if any(label.source in REPORTED for label in labels.get(top.run_id, []))
            ]
        under = children_of(rows)
        return [(read_back(top, under), labels.get(top.run_id, [])) for top in tops[:limit]]

    async def delete_scope(self, scope: str) -> int:
        if not scope:
            raise InputError("refinery: scope is required")
        return await self.storage.delete_scope(scope)

    async def delete_runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> int:
        if not scope:
            raise InputError("refinery: scope is required")
        if run_ids is None and tags is None:
            raise InputError("refinery: delete_runs takes run_ids or tags")
        check_selector("delete_runs", run_ids, tags)
        return await self.storage.delete_runs(scope, run_ids=run_ids, tags=tags)

    async def verify_absent(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> bool:
        if not scope:
            raise InputError("refinery: scope is required")
        if run_ids is None and tags is None:
            return await self.storage.count_scope(scope) == 0
        check_selector("verify_absent", run_ids, tags)
        if await self.storage.run_tree(scope, run_ids=run_ids, tags=tags):
            return False
        return run_ids is None or not await self.storage.labels(scope, run_ids=run_ids)
