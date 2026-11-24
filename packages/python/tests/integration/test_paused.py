import re
from pathlib import Path
from typing import Any

import pytest
from unit.fakes import Crew, background, delegating, team, text_reply
from unit.test_approve import Refunds
from unit.test_paused import REFUND, calling, found, later_for
from unit.trees import write_folder

from integration.stores import PostgresStore, held
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


def paused_team(root: Path, storage: Storage) -> Engine:
    both = Reply(
        tool_calls=[
            *background("part one").tool_calls,
            *delegating("desk", "refund it", call_id="d9").tool_calls,
        ],
        stop_reason="tool_use",
    )
    model = Crew(
        [both],
        {"part one": [calling(LOOKUP)], "refund it": [calling(REFUND)]},
    )
    return team(root, model, refinery=Refinery(storage))


def resumed_team(root: Path, storage: Storage) -> Engine:
    model = Crew(
        [text_reply("waiting"), text_reply("all done")],
        {"part one": [text_reply("part one done")], "refund it": [text_reply("refunded")]},
    )
    return team(root, model, refinery=Refinery(storage))


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
    first_engine = paused_team(root, storage)
    refunds = Refunds()
    first = await start(first_engine, store, refunds)
    assert {w.agent for w in first.paused} == {"worker", "desk"} and refunds.done == []
    await first_engine.close()
    after = Refunds()
    second = resumed_team(root, storage)
    try:
        resumed = await start(second, store, after, answers={w.id: True for w in first.paused})
    finally:
        await second.close()
    assert resumed.error is None and resumed.paused == [] and resumed.resumed
    assert resumed.reply == "all done"
    assert refunds.done == [] and after.done == [{"amount": 10}]
    rows = {r.run_id: r for r in await storage.run_tree("acme", tags={"workflow": "w1"})}
    assert resumed.run_id in rows
    again = [c for c in resumed.children if isinstance(c, Generation)]
    assert {(c.agent, c.reply) for c in again} == {
        ("worker", "part one done"),
        ("desk", "refunded"),
    }
    for child in again:
        assert child.error is None and child.paused == [] and child.resumed
        row = rows[child.run_id]
        assert (row.agent, row.body["reply"], row.body["error"]) == (child.agent, child.reply, None)
    conversations = {"c1", *(str(w.conversation) for w in first.paused)}
    for name in conversations:
        last = (await held(store, "acme", name))[-1]
        assert last.role == "assistant" and last.pause is None and last.content != "[paused]"


async def test_a_paused_conversation_stays_in_its_scope(
    tmp_path: Path, store: PostgresStore, storage: Storage
) -> None:
    root = folder(tmp_path)
    engine = paused_team(root, storage)
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
