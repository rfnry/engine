import re
from pathlib import Path
from typing import Any

import pytest
from unit.fakes import Crew, background, delegating, team, text_reply
from unit.test_approve import Refunds
from unit.test_paused import REFUND, calling, found, later_for
from unit.trees import write_folder

from integration.stores import PostgresStore
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.tools import Tool, Tools

LOOKUP = ToolCall(id="c-lookup", name="lookup", args={})


def folder(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    root.mkdir(parents=True)
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the work",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["worker", "desk"], "background": True},
            },
            "worker": {
                "kind": "generation",
                "model": "crew",
                "description": "Looks orders up",
                "instructions": "worker.md",
                "tools": {"allow": ["lookup"]},
            },
            "desk": {
                "kind": "generation",
                "model": "crew",
                "description": "Refunds orders",
                "instructions": "desk.md",
                "tools": {"allow": ["refund"]},
            },
        },
        runtime_tools=["lookup", "refund"],
        files={"lead.md": "Lead.\n", "worker.md": "Work.\n", "desk.md": "Refund.\n"},
    )


def tools(refunds: Refunds) -> Tools:
    return Tools([Tool("lookup", "Look an order up", body=found), refunds.tools()])


def paused_team(root: Path, storage: Storage) -> tuple[Engine, Crew, Refinery]:
    both = Reply(
        tool_calls=[
            *background("part one").tool_calls,
            *delegating("desk", "refund it", call_id="d9").tool_calls,
        ],
        stop_reason="tool_use",
    )
    model = Crew(
        [both, text_reply("waiting"), text_reply("all done")],
        {
            "part one": [calling(LOOKUP), text_reply("part one done")],
            "refund it": [calling(REFUND), text_reply("refunded")],
        },
    )
    refinery = Refinery(storage)
    return team(root, model, refinery=refinery), model, refinery


async def start(
    engine: Engine, store: PostgresStore, refunds: Refunds, **options: Any
) -> Generation:
    run = await engine.run(
        "lead",
        "acme",
        "go",
        store=store,
        conversation="c1",
        tools=tools(refunds),
        approve=later_for("lookup", "refund"),
        tags={"workflow": "w1"},
        **options,
    )
    assert isinstance(run, Generation)
    return run


async def test_a_paused_team_resumes_in_a_new_engine(
    tmp_path: Path, store: PostgresStore, storage: Storage
) -> None:
    root = folder(tmp_path)
    first_engine, model, refinery = paused_team(root, storage)
    refunds = Refunds()
    first = await start(first_engine, store, refunds)
    assert {w.agent for w in first.paused} == {"worker", "desk"} and refunds.done == []
    await first_engine.close()
    second = team(root, model, refinery=refinery)
    try:
        resumed = await start(second, store, refunds, answers={w.id: True for w in first.paused})
    finally:
        await second.close()
    assert resumed.error is None and resumed.paused == [] and resumed.resumed
    assert refunds.done == [{"amount": 10}]
    rows = await storage.run_tree("acme", tags={"workflow": "w1"})
    assert {r.agent for r in rows} >= {"lead", "worker", "desk"}


async def test_a_paused_conversation_stays_in_its_scope(
    tmp_path: Path, store: PostgresStore, storage: Storage
) -> None:
    root = folder(tmp_path)
    engine, model, _ = paused_team(root, storage)
    refunds = Refunds()
    try:
        first = await start(engine, store, refunds)
        [wid, *_] = [w.id for w in first.paused]
        with pytest.raises(
            InputError, match=re.escape(f"engine: no call waits for an answer under id {wid!r}")
        ):
            await engine.run(
                "lead",
                "other",
                "go",
                store=store,
                conversation="c1",
                tools=tools(refunds),
                answers={wid: True},
            )
        assert await store.length("other", "c1") == 0
        assert refunds.done == []
    finally:
        await engine.close()
