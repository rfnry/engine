import asyncio
import base64
import contextlib
import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, JsonValue

from rfnry_engine import Engine, tree
from rfnry_engine.errors import ConfigurationError, InputError, WorkerError
from rfnry_engine.models import Image, Message, Reply, Text, ToolCall
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace
from rfnry_engine.tree import load_tree

from .fakes import Answering, Scripted, text_reply

RUN = "def run(args, call):\n    return {}\n"
PID = "import os, time\ndef run(args, call):\n    time.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
ORDER: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def tool(
    folder: Path,
    name: str,
    description: str = "Do it",
    body: str = RUN,
    input_schema: dict[str, JsonValue] | None = None,
    output_schema: dict[str, JsonValue] | None = None,
) -> None:
    definition: dict[str, JsonValue] = {
        "description": description,
        "input_schema": {"type": "object"} if input_schema is None else input_schema,
    }
    if output_schema is not None:
        definition["output_schema"] = output_schema
    write(folder / f"{name}.json", json.dumps(definition))
    write(folder / f"{name}.py", body)


def calling(*calls: tuple[str, dict[str, Any]]) -> Reply:
    return Reply(
        tool_calls=[
            ToolCall(id=f"c{n}", name=name, args=args) for n, (name, args) in enumerate(calls, 1)
        ],
        stop_reason="tool_use",
    )


def pid(value: object) -> int:
    assert isinstance(value, int)
    return value


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


async def gone(pid: int) -> bool:
    for _ in range(40):
        if not alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def pong(args: dict[str, Any]) -> str:
    return "pong"


async def generated(engine: Engine, *args: Any, **options: Any) -> Generation:
    run = await engine.run(*args, **options)
    assert isinstance(run, Generation)
    return run


def shop(root: Path, extra: str = "") -> Path:
    write(root / "shop/AGENT.md", agent_md("shop", extra=extra))
    return root


async def test_a_model_calls_a_folder_tool_and_reads_its_result(tmp_path: Path) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop"))
    tool(
        tmp_path / "shop/tools",
        "lookup_order",
        body="def run(args, call):\n    return {'order': args['order_id'], 'scope': call['scope']}\n",
        input_schema=ORDER,
    )
    llm = Scripted(
        Reply(
            tool_calls=[ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})],
            stop_reason="tool_use",
        ),
        text_reply("done"),
    )
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
    assert isinstance(run.steps[0].result, str)
    assert json.loads(run.steps[0].result) == {"order": "A-1", "scope": "acme"}
    assert not run.steps[0].is_error


async def test_a_script_reads_the_call_it_serves(tmp_path: Path) -> None:
    tool(shop(tmp_path) / "shop/tools", "whose", body="def run(args, call):\n    return call\n")
    llm = Scripted(calling(("whose", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
        outside = await EngineTools(engine, agent="shop", scope="globex").dispatch("whose", {})
    assert isinstance(run.steps[0].result, str)
    assert json.loads(run.steps[0].result) == {
        "run_id": run.run_id,
        "agent": "shop",
        "scope": "acme",
        "tool": "whose",
        "call_id": "c1",
        "by": None,
        "tags": {},
    }
    assert outside == {"scope": "globex", "tags": {}}


WHOSE = "def run(args, call):\n    return {'agent': call.get('agent'), 'tags': call['tags']}\n"


async def test_a_script_reads_the_tags_of_the_run_it_serves(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "whose", body=WHOSE)
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="agents: [shop]\n"))
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra="description: Looks things up\n"))
    write(
        tmp_path / "desk/AGENT.md",
        agent_md(
            "desk", kind="decision", model="judge", extra="route: {by: go, to: {next: shop}}\n"
        ),
    )
    write(
        tmp_path / "desk/questions/go.json",
        '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep", "next": "Pass"}}',
    )
    ticket = {"ticket": "T-7"}
    llm = Scripted(
        calling(("whose", {})),
        text_reply("own"),
        calling(("Delegate", {"agent": "shop", "input": "look"})),
        calling(("whose", {})),
        text_reply("child"),
        text_reply("lead"),
        calling(("whose", {})),
        text_reply("routed"),
        calling(("whose", {})),
        text_reply("consulted"),
    )
    recorded: list[Run] = []

    class Recording:
        async def record(self, run: Run) -> None:
            recorded.append(run)

    async with Engine(
        tmp_path, models={"strong": llm, "judge": Answering()}, refinery=Recording()
    ) as engine:
        own = await generated(engine, "shop", "acme", "x", tags=ticket)
        lead = await generated(engine, "lead", "acme", "x", tags=ticket)
        routed = await engine.run("desk", "acme", "x", tags=ticket)
        consult = engine.tool("shop", scope="acme", tags={"ticket": "T-9"})
        await consult.dispatch("shop", {"input": "look"})
        outside = await EngineTools(engine, agent="shop", scope="acme", tags=ticket).dispatch(
            "whose", {}
        )
    [child] = lead.children
    assert isinstance(child, Generation) and isinstance(routed, Decision)
    assert isinstance(routed.routed_to, Generation)
    consulted = recorded[-1]
    assert isinstance(consulted, Generation) and consulted.tags == {"ticket": "T-9"}
    assert [
        json.loads(str(step.result))
        for step in (own.steps[0], child.steps[0], routed.routed_to.steps[0], consulted.steps[0])
    ] == [{"agent": "shop", "tags": ticket}] * 3 + [{"agent": "shop", "tags": {"ticket": "T-9"}}]
    assert outside == {"agent": None, "tags": ticket}


async def test_invalid_arguments_never_reach_the_worker(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    body = f"from pathlib import Path\ndef run(args, call):\n    Path({str(marker)!r}).write_text('ran')\n    return 'ok'\n"
    tool(shop(tmp_path) / "shop/tools", "lookup_order", body=body, input_schema=ORDER)
    llm = Scripted(calling(("lookup_order", {"order_id": 5})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
        with pytest.raises(InputError, match="is a required property"):
            await EngineTools(engine, agent="shop", scope="acme").dispatch("lookup_order", {})
    assert run.steps[0].is_error
    assert run.steps[0].result == (
        'InputError: tools/declared: invalid arguments for lookup_order: 5 is not of type "string" at order_id'
    )
    assert not marker.exists()


async def test_arguments_from_code_that_json_cannot_hold_never_reach_the_worker(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"
    body = f"from pathlib import Path\ndef run(args, call):\n    Path({str(marker)!r}).write_text('ran')\n    return 'ok'\n"
    tool(shop(tmp_path) / "shop/tools", "note", body=body)
    async with Engine(tmp_path, models={}) as engine:
        surface = EngineTools(engine, agent="shop", scope="acme")
        with pytest.raises(InputError) as raised:
            await surface.dispatch("note", {"when": datetime.datetime(2026, 1, 1)})
        assert str(raised.value) == (
            "workers/pool: the arguments for note are not JSON: Object of type datetime is not JSON serializable"
        )
        assert not marker.exists()
        assert await surface.dispatch("note", {}) == "ok"
        pool = engine._workers.pool("shop")
        assert [len(worker.calls) for worker in pool._workers if worker is not None] == [0]


async def test_a_result_off_its_output_schema_is_a_tool_error(tmp_path: Path) -> None:
    tool(
        shop(tmp_path) / "shop/tools",
        "lookup_order",
        output_schema={"type": "object", "required": ["status"]},
    )
    llm = Scripted(calling(("lookup_order", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
    assert run.steps[0].is_error
    assert isinstance(run.steps[0].result, str)
    assert run.steps[0].result.startswith(
        "ModelError: tools/declared: lookup_order returned a result that does not match its output_schema: "
        '"status" is a required property'
    )


async def test_the_model_reads_a_folder_tool_by_its_name_description_and_input_schema_only(
    tmp_path: Path,
) -> None:
    tool(
        shop(tmp_path) / "shop/tools",
        "lookup_order",
        description="Fetch an order by its id",
        input_schema=ORDER,
        output_schema={"type": "object"},
    )
    llm = Scripted(text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        await engine.run("shop", "acme", "x")
        [shown] = [
            d
            for d in EngineTools(engine, agent="shop", scope="acme").definitions()
            if d["name"] == "lookup_order"
        ]
    assert shown == {
        "name": "lookup_order",
        "description": "Fetch an order by its id",
        "input_schema": ORDER,
    }
    [(_, _, sent)] = llm.calls
    [spec] = [t for t in sent if t.name == "lookup_order"]
    assert spec.model_dump() == shown


async def test_a_result_of_parts_reaches_the_model_as_parts_and_other_json_as_it_is(
    tmp_path: Path,
) -> None:
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    parts = f"def run(args, call):\n    return [{{'type': 'text', 'text': 'the label'}}, {{'type': 'image', 'media_type': 'image/png', 'data': {png!r}}}]\n"
    listed = "def run(args, call):\n    return [{'type': 'order', 'id': 1}, 2]\n"
    tool(shop(tmp_path) / "shop/tools", "label", body=parts)
    tool(tmp_path / "shop/tools", "orders", body=listed)
    llm = Scripted(calling(("label", {}), ("orders", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
    assert run.steps[0].result == [
        Text(text="the label"),
        Image(media_type="image/png", data=b"\x89PNG\r\n\x1a\n"),
    ]
    assert run.steps[1].result == '[{"type": "order", "id": 1}, 2]'


async def test_parts_that_do_not_fit_their_contract_are_a_tool_error(tmp_path: Path) -> None:
    body = "def run(args, call):\n    return [{'type': 'image', 'data': 'aGk='}]\n"
    tool(shop(tmp_path) / "shop/tools", "label", body=body)
    llm = Scripted(calling(("label", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
    assert run.steps[0].is_error and isinstance(run.steps[0].result, str)
    assert run.steps[0].result.startswith(
        "ModelError: tools/declared: label returned parts that do not fit the contract: "
    )
    assert "media_type" in run.steps[0].result


async def test_a_script_that_raises_is_a_tool_error_the_model_reads(tmp_path: Path) -> None:
    body = "def run(args, call):\n    raise ValueError(f\"no order {args['order_id']}\")\n"
    tool(shop(tmp_path) / "shop/tools", "lookup_order", body=body, input_schema=ORDER)
    llm = Scripted(calling(("lookup_order", {"order_id": "A-1"})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
    assert run.steps[0].is_error
    assert run.steps[0].result == "WorkerError: ValueError: no order A-1"


async def test_a_crashing_script_is_a_tool_error_and_the_run_goes_on(tmp_path: Path) -> None:
    tool(
        shop(tmp_path) / "shop/tools",
        "crash",
        body="import os\ndef run(args, call):\n    os._exit(3)\n",
    )
    tool(tmp_path / "shop/tools", "fine", body="def run(args, call):\n    return 'fine'\n")
    llm = Scripted(calling(("crash", {})), calling(("fine", {})), text_reply("carried on"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x")
    assert run.steps[0].is_error
    assert (
        run.steps[0].result
        == "WorkerError: workers/pool: the worker for crash stopped: exited with code 3"
    )
    assert (run.steps[1].result, run.steps[1].is_error) == ("fine", False)
    assert run.reply == "carried on"


async def test_a_folder_tool_s_result_is_held_to_the_engine_s_limit(tmp_path: Path) -> None:
    body = "def run(args, call):\n    return 'x' * args['size']\n"
    tool(shop(tmp_path) / "shop/tools", "big", body=body)
    llm = Scripted(calling(("big", {"size": 5000})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}, worker_result_bytes=1000) as engine:
        run = await generated(engine, "shop", "acme", "x")
        surface = EngineTools(engine, agent="shop", scope="acme")
        assert await surface.dispatch("big", {"size": 100}) == "x" * 100
    assert run.steps[0].is_error
    assert re.fullmatch(
        r"WorkerError: workers/process: big sent back \d+ bytes, over the limit of 1000 bytes",
        str(run.steps[0].result),
    )
    assert run.reply == "done"


async def test_a_folder_tool_owns_its_deadline_inside_a_run_and_from_code(tmp_path: Path) -> None:
    body = "import asyncio\nasync def run(args, call):\n    await asyncio.sleep(30)\n"
    tool(shop(tmp_path) / "shop/tools", "slow", body=body)
    llm = Scripted(calling(("slow", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}, tool_timeout=0.5) as engine:
        run = await generated(engine, "shop", "acme", "x")
        with pytest.raises(TimeoutError, match=r"^workers/pool: slow timed out after 0\.5s$"):
            await EngineTools(engine, agent="shop", scope="acme").dispatch("slow", {})
    assert run.steps[0].is_error
    assert run.steps[0].result == "TimeoutError: workers/pool: slow timed out after 0.5s"


async def test_an_allow_list_keeps_only_the_named_folder_and_file_tools(tmp_path: Path) -> None:
    write(
        tmp_path / "shop/AGENT.md",
        agent_md("shop", extra="agents: [aide]\nworkspace: true\ntools: [lookup_order, Read]\n"),
    )
    write(tmp_path / "aide/AGENT.md", agent_md("aide", extra="description: Helps\n"))
    tool(tmp_path / "shop/tools", "lookup_order")
    tool(tmp_path / "shop/tools", "notify")
    (tmp_path / "work").mkdir()
    ping = Tools([Tool("ping", "Ping", body=pong)])
    async with (
        Engine(tmp_path, models={}) as engine,
        Workspace(Local(tmp_path / "work"), scope="acme") as room,
    ):
        names = EngineTools(
            engine,
            agent="shop",
            scope="acme",
            tools=ping,
            workspace=room,
            history=[Message(role="user", content="hi")],
            start=1,
        ).names
    assert sorted(names) == sorted(
        ["Read", "History", "lookup_order", "Shell", "Job", "Jobs", "Write", "Edit", "Delegate"]
    )
    assert engine.agents["shop"].tools == ["lookup_order"]


async def test_an_allow_list_keeps_only_the_named_run_tools(tmp_path: Path) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra="tools: [ping, lookup_order, Read]\n"))
    tool(tmp_path / "shop/tools", "lookup_order")
    given = Tools([Tool("ping", "Ping", body=pong), Tool("pong", "Pong", body=pong)])
    llm = Scripted(calling(("ping", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x", tools=given)
        names = EngineTools(engine, agent="shop", scope="acme", tools=given).names
    assert sorted(names) == ["Read", "lookup_order", "ping"]
    assert sorted(t.name for t in run.tools) == ["Read", "lookup_order", "ping"]
    assert (run.steps[0].result, run.steps[0].is_error) == ("pong", False)
    assert engine.agents["shop"].tools == ["lookup_order"]


async def test_a_delegated_or_routed_agent_keeps_only_the_inherited_run_tools_it_names(
    tmp_path: Path,
) -> None:
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="agents: [drafter, helper]\n"))
    write(
        tmp_path / "drafter/AGENT.md",
        agent_md("drafter", extra="description: Drafts\ntools: [Read]\n"),
    )
    write(tmp_path / "helper/AGENT.md", agent_md("helper", extra="description: Helps\n"))
    write(
        tmp_path / "desk/AGENT.md",
        agent_md(
            "desk", kind="decision", model="judge", extra="route: {by: go, to: {next: drafter}}\n"
        ),
    )
    write(
        tmp_path / "desk/questions/go.json",
        '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep", "next": "Pass"}}',
    )
    given = Tools([Tool("remember", "Remember", body=pong)])
    llm = Scripted(
        calling(("Delegate", {"agent": "drafter", "input": "draft"})),
        text_reply("drafted"),
        calling(("Delegate", {"agent": "helper", "input": "help"})),
        text_reply("helped"),
        text_reply("lead done"),
        text_reply("routed"),
    )
    async with Engine(tmp_path, models={"strong": llm, "judge": Answering()}) as engine:
        lead = await generated(engine, "lead", "acme", "x", tools=given)
        routed = await engine.run("desk", "acme", "x", tools=given)
    drafter, helper = lead.children
    assert isinstance(drafter, Generation) and isinstance(helper, Generation)
    assert isinstance(routed, Decision) and isinstance(routed.routed_to, Generation)
    assert "remember" in [t.name for t in lead.tools]
    assert [t.name for t in drafter.tools] == ["Read"]
    assert "remember" in [t.name for t in helper.tools]
    assert [t.name for t in routed.routed_to.tools] == ["Read"]
    assert drafter.fingerprint.tools == routed.routed_to.fingerprint.tools


@pytest.mark.parametrize(
    "written", ["refnd", '"!refund"', 'refund, "!Grep"', '"!Delegate"', '"!Output"', "Delegate"]
)
async def test_a_kept_name_no_folder_gives_loads_and_a_run_that_does_not_offer_it_is_refused(
    tmp_path: Path, written: str
) -> None:
    write(tmp_path / "support/AGENT.md", agent_md("support", extra=f"tools: [{written}]\n"))
    tool(tmp_path / "support/tools", "refund")
    kept = [name.strip().strip('"') for name in written.split(",")][-1]
    llm = Scripted()
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        with pytest.raises(ConfigurationError) as caught:
            await engine.run("support", "acme", "x")
        with pytest.raises(ConfigurationError) as surfaced:
            EngineTools(engine, agent="support", scope="acme")
        given = Tools([Tool(kept, "Given", body=pong)])
        offered = EngineTools(engine, agent="support", scope="acme", tools=given).names
    said = (
        f"tools/engine_tools: agent 'support' keeps '{kept}' in tools:, which this run does not "
        "offer; its tools are Glob, Grep, LS, Load, Read, refund"
    )
    assert str(caught.value) == str(surfaced.value) == said
    assert llm.calls == []
    assert kept in offered


async def test_a_delegated_run_that_does_not_offer_a_kept_name_is_a_tool_error_and_a_route_raises(
    tmp_path: Path,
) -> None:
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="agents: [drafter]\n"))
    write(
        tmp_path / "drafter/AGENT.md",
        agent_md("drafter", extra="description: Drafts\ntools: [remember]\n"),
    )
    write(
        tmp_path / "desk/AGENT.md",
        agent_md(
            "desk", kind="decision", model="judge", extra="route: {by: go, to: {next: drafter}}\n"
        ),
    )
    write(
        tmp_path / "desk/questions/go.json",
        '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep", "next": "Pass"}}',
    )
    llm = Scripted(
        calling(("Delegate", {"agent": "drafter", "input": "draft"})), text_reply("done")
    )
    said = (
        "tools/engine_tools: agent 'drafter' keeps 'remember' in tools:, which this run does not "
        "offer; its tools are Glob, Grep, LS, Load, Read"
    )
    async with Engine(tmp_path, models={"strong": llm, "judge": Answering()}) as engine:
        lead = await generated(engine, "lead", "acme", "x")
        with pytest.raises(ConfigurationError) as caught:
            await engine.run("desk", "acme", "x")
    assert (lead.steps[0].result, lead.steps[0].is_error) == (f"ConfigurationError: {said}", True)
    assert str(caught.value) == said


async def test_an_allow_list_leaves_output_to_expect(tmp_path: Path) -> None:
    class Verdict(BaseModel):
        ok: bool

    tool(shop(tmp_path, "tools: [lookup_order]\n") / "shop/tools", "lookup_order")
    llm = Scripted(calling(("Output", {"ok": True})))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x", expect=Verdict)
    [(_, _, sent)] = llm.calls
    assert [t.name for t in sent] == ["lookup_order", "Output"]
    assert run.reply == '{"ok":true}'


async def test_a_run_tool_named_like_a_folder_tool_wins(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    body = f"from pathlib import Path\ndef run(args, call):\n    Path({str(marker)!r}).write_text('ran')\n    return 'script'\n"
    tool(
        shop(tmp_path) / "shop/tools",
        "lookup_order",
        description="The folder's",
        body=body,
        input_schema=ORDER,
    )

    async def stub(args: dict[str, Any]) -> str:
        return f"stub {args['order_id']}"

    stubbed = Tools([Tool("lookup_order", "The run's", body=stub)])
    llm = Scripted(calling(("lookup_order", {"order_id": 5})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        run = await generated(engine, "shop", "acme", "x", tools=stubbed)
        surface = EngineTools(engine, agent="shop", scope="acme", tools=stubbed)
    assert (run.steps[0].result, run.steps[0].is_error) == ("stub 5", False)
    assert not marker.exists()
    _, _, sent = llm.calls[0]
    assert [t.description for t in sent if t.name == "lookup_order"] == ["The run's"]
    assert surface.names.count("lookup_order") == 1


async def test_two_run_tools_with_one_name_are_refused_whatever_tools_keeps(
    tmp_path: Path,
) -> None:
    class Doubled:
        def definitions(self) -> list[dict[str, Any]]:
            return [Tool("ping", "Ping", body=pong).definition()] * 2

        async def dispatch(self, name: str, args: dict[str, Any]) -> str:
            return "pong"

    tool(shop(tmp_path, "tools: [lookup_order]\n") / "shop/tools", "lookup_order")
    async with Engine(tmp_path, models={}) as engine:
        with pytest.raises(ConfigurationError, match="^tools: duplicate tool name 'ping'$"):
            EngineTools(engine, agent="shop", scope="acme", tools=Doubled())


async def test_a_run_tool_named_like_a_folder_tool_keeps_the_run_s_deadline(
    tmp_path: Path,
) -> None:
    tool(shop(tmp_path) / "shop/tools", "slow")

    async def stalls(args: dict[str, Any]) -> str:
        await asyncio.sleep(30)
        return "never"

    llm = Scripted(calling(("slow", {})), text_reply("done"))
    async with Engine(tmp_path, models={"strong": llm}, tool_timeout=0.2) as engine:
        run = await generated(
            engine, "shop", "acme", "x", tools=Tools([Tool("slow", "Slow", body=stalls)])
        )
    assert run.steps[0].result == "TimeoutError: engine: slow timed out after 0.2s"


@pytest.mark.parametrize(
    ("tools", "said"),
    [
        (
            "[Output]",
            "tools: 'Output' cannot be kept by name; it comes with expect= on a run, so leave it out of tools:",
        ),
        (
            "[' x']",
            "tools: an entry must not be blank or padded, got ' x'; write the name of a tool to keep, such as tools: [lookup_order]",
        ),
        ("['']", "tools: an entry must not be blank or padded"),
        ("[Read, Read]", "tools: 'Read' is listed twice; list each tool once"),
        ("[3]", "tools"),
        (
            "",
            "tools: it is empty; write tools: all to keep every tool, tools: none to keep none, or the tools to keep, such as tools: [lookup_order]",
        ),
    ],
)
def test_a_tools_list_that_breaks_the_grammar_is_refused_naming_the_file(
    tmp_path: Path, tools: str, said: str
) -> None:
    shop(tmp_path, f"tools: {tools}\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(tmp_path / "shop/AGENT.md") in str(caught.value) and said in str(caught.value)


@pytest.mark.parametrize("tools", ["[x]", "all", "none"])
def test_a_decision_agent_cannot_list_tools(tmp_path: Path, tools: str) -> None:
    write(
        tmp_path / "triage/AGENT.md",
        agent_md("triage", kind="decision", model="judge", extra=f"tools: {tools}\n"),
    )
    with pytest.raises(ConfigurationError, match="decision agent 'triage' cannot list tools:"):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    ("listed", "kept"),
    [
        ("", ["Read", "Grep", "Glob", "LS", "Load", "lookup_order", "ping"]),
        ("tools: none\n", []),
        ("tools: all\n", ["Read", "Grep", "Glob", "LS", "Load", "lookup_order", "ping"]),
    ],
)
async def test_an_empty_tools_keeps_no_folder_file_or_run_tool_and_none_keeps_them_all(
    tmp_path: Path, listed: str, kept: list[str]
) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra=f"agents: [aide]\n{listed}"))
    write(tmp_path / "aide/AGENT.md", agent_md("aide", extra="description: Helps\n"))
    tool(tmp_path / "shop/tools", "lookup_order")
    ping = Tools([Tool("ping", "Ping", body=pong)])
    async with Engine(tmp_path, models={}) as engine:
        names = EngineTools(
            engine,
            agent="shop",
            scope="acme",
            tools=ping,
            history=[Message(role="user", content="hi")],
            start=1,
        ).names
    assert sorted(names) == sorted([*kept, "History", "Delegate"])
    assert engine.agents["shop"].tools == [name for name in kept if name == "lookup_order"]


@pytest.mark.parametrize(
    ("written", "extra"),
    [
        ("Delegate", "agents: [aide]\n"),
        ("History", ""),
        ("Shell", "workspace: true\n"),
        ("refund, Edit", "workspace: true\n"),
    ],
)
def test_a_kept_name_of_a_tool_that_comes_with_its_own_declaration_is_refused(
    tmp_path: Path, written: str, extra: str
) -> None:
    write(tmp_path / "support/AGENT.md", agent_md("support", extra=f"{extra}tools: [{written}]\n"))
    write(tmp_path / "aide/AGENT.md", agent_md("aide"))
    tool(tmp_path / "support/tools", "refund")
    kept = written.split(", ")[-1]
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: agent 'support' keeps {kept!r} in tools:, which comes with its own declaration "
        "and stays whatever tools: lists; leave it out of tools:"
    )


def test_a_leftover_tools_md_is_content_and_a_markdown_file_in_tools_is_no_tool(
    tmp_path: Path,
) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra="preload: ['.']\n"))
    write(tmp_path / "shop/TOOLS.md", "---\nfetch: Fetch\n---\n")
    write(tmp_path / "shop/tools/fetch.md", "---\ndescription: Fetch\n---\nHow to fetch.\n")
    spec = load_tree(tmp_path)["shop"]
    assert (spec.keep, spec.folder_tools) == (None, {})
    assert spec.listing == "TOOLS.md"
    sent = "".join(str(part) for part in spec.preload)
    assert "fetch: Fetch" in sent and "How to fetch." not in sent


def test_the_tools_an_agent_keeps_are_part_of_its_fingerprint(tmp_path: Path) -> None:
    before = load_tree(shop(tmp_path, "tools: [Read]\n"))["shop"].fingerprint
    after = load_tree(shop(tmp_path, "tools: [Read, Grep]\n"))["shop"].fingerprint
    assert before != after


def test_a_kept_name_may_be_a_tool_of_a_folder_above(tmp_path: Path) -> None:
    write(tmp_path / "desk/support/AGENT.md", agent_md("support", extra="tools: [notify]\n"))
    tool(tmp_path / "desk/tools", "notify")
    assert list(load_tree(tmp_path)["support"].folder_tools) == ["notify"]


async def test_a_tool_of_a_folder_nearer_the_agent_replaces_one_of_the_same_name_above(
    tmp_path: Path,
) -> None:
    for place, said in (("tools", "root"), ("team/tools", "team")):
        tool(
            tmp_path / place,
            "notify",
            description=f"{said} notify",
            body=f"def run(args, call):\n    return 'from {said}'\n",
        )
    write(tmp_path / "team/writer/AGENT.md", agent_md("writer"))
    write(tmp_path / "desk/AGENT.md", agent_md("desk"))
    async with Engine(tmp_path, models={}) as engine:
        got = {}
        for agent in ("writer", "desk"):
            surface = EngineTools(engine, agent=agent, scope="acme")
            [notify] = [d for d in surface.definitions() if d["name"] == "notify"]
            got[agent] = (notify["description"], await surface.dispatch("notify", {}))
    assert got == {
        "writer": ("team notify", "from team"),
        "desk": ("root notify", "from root"),
    }


async def test_workers_puts_a_folder_s_agents_on_its_own_processes(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "pid", body=PID)
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    write(tmp_path / "support/help/AGENT.md", agent_md("help"))
    write(tmp_path / "desk/AGENT.md", agent_md("desk"))
    async with Engine(tmp_path, models={}, workers={"support": 2}) as engine:

        async def pids(agent: str, n: int) -> set[int]:
            surface = EngineTools(engine, agent=agent, scope="acme")
            got = await asyncio.gather(*(surface.dispatch("pid", {"sleep": 0.3}) for _ in range(n)))
            return {pid(value) for value in got}

        support, desk = await pids("billing", 2), await pids("desk", 2)
        helped = await pids("help", 1)
    assert len(support) == 2 and len(desk) == 1
    assert helped <= support and not support & desk


@pytest.mark.parametrize("cut", ["while-a-worker-spawns", "as-it-starts"])
async def test_a_cancelled_close_still_stops_every_worker_of_a_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    tool(tmp_path / "tools", "pid", body=PID)
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    engine = Engine(tmp_path, models={}, workers={"support": 2})
    surface = EngineTools(engine, agent="billing", scope="acme")
    pool = engine._workers.pool("billing")
    calls: list[asyncio.Future[Any]] = []
    try:
        async with asyncio.timeout(10):
            first = pid(await surface.dispatch("pid", {}))
            calls.append(asyncio.ensure_future(surface.dispatch("pid", {"sleep": 30})))
            spawning, release = asyncio.Event(), asyncio.Event()
            spawn = pool._spawn

            async def held() -> Any:
                spawning.set()
                await release.wait()
                return await spawn()

            if cut == "while-a-worker-spawns":
                monkeypatch.setattr(pool, "_spawn", held)
                calls.append(asyncio.ensure_future(surface.dispatch("pid", {})))
                await spawning.wait()
            closing = asyncio.ensure_future(engine.close())
            await asyncio.sleep(0 if cut == "as-it-starts" else 0.05)
            closing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await closing
            release.set()
            await asyncio.wait(calls)
        assert await gone(first)
        assert all(isinstance(call.exception(), WorkerError) for call in calls)
    finally:
        await engine.close()
        for call in calls:
            if not call.done():
                call.cancel()


async def test_workers_holds_a_script_once_per_pool_however_many_agents_see_it(
    tmp_path: Path,
) -> None:
    tool(tmp_path / "tools", "pid", body=PID)
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    write(tmp_path / "support/help/AGENT.md", agent_md("help"))
    async with Engine(tmp_path, models={}, workers={"support": 1}) as engine:
        pool = engine._workers.pool("billing")
        assert pool is engine._workers.pool("help")
    assert pool._scripts["scripts"] == [
        {"key": "tools/pid", "path": str((tmp_path / "tools/pid.py").resolve()), "source": PID}
    ]


async def test_two_tools_folders_of_one_worker_group_hold_a_tool_of_one_name_each(
    tmp_path: Path,
) -> None:
    for side in ("left", "right"):
        tool(
            tmp_path / "shop" / side / "tools",
            "who",
            body=f"def run(args, call):\n    return '{side}'\n",
        )
        write(tmp_path / "shop" / side / "agent/AGENT.md", agent_md(f"{side}_agent"))
    async with Engine(tmp_path, models={}, workers={"shop": 1}) as engine:
        pool = engine._workers.pool("left_agent")
        assert pool is engine._workers.pool("right_agent")
        got = [
            await EngineTools(engine, agent=agent, scope="acme").dispatch("who", {})
            for agent in ("left_agent", "right_agent", "left_agent")
        ]
    assert got == ["left", "right", "left"]
    assert [item["key"] for item in cast(Any, pool._scripts)["scripts"]] == [
        "shop/left/tools/who",
        "shop/right/tools/who",
    ]


async def test_a_worker_tells_its_pool_which_scripts_take_a_thread(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "pid", body=PID)
    tool(tmp_path / "tools", "wait", body="async def run(args, call):\n    return 1\n")
    write(tmp_path / "shop/AGENT.md", agent_md("shop"))
    write(tmp_path / "desk/AGENT.md", agent_md("desk", kind="decision", model="judge"))
    for name, kind in (("order", "async def"), ("size", "def")):
        write(tmp_path / f"desk/questions/{name}.json", '{"type": "choice", "instructions": "?"}')
        write(
            tmp_path / f"desk/questions/{name}.py",
            f"{kind} run(input, call):\n    return {{'options': {{'a': 'A', 'b': 'B'}}}}\n",
        )
    async with Engine(tmp_path, models={}) as engine:
        assert pid(await EngineTools(engine, agent="shop", scope="acme").dispatch("wait", {})) == 1
        [worker] = engine._workers.pool("shop")._workers
        assert worker is not None and worker.threaded == {
            "tools/pid:run",
            "desk/questions/size:run",
        }


async def test_closing_the_engine_reports_no_worker_exit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    tool(tmp_path / "tools", "pid", body=PID)
    write(tmp_path / "shop/AGENT.md", agent_md("shop"))
    async with Engine(tmp_path, models={}) as engine:
        worker = pid(await EngineTools(engine, agent="shop", scope="acme").dispatch("pid", {}))
    assert not Path(f"/proc/{worker}").exists()
    await asyncio.sleep(0.2)
    assert [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ] == []


@pytest.mark.parametrize(
    ("workers", "said"),
    [
        (
            {"billing": 1},
            "engine: workers names 'billing', which holds no agent of the tree; name a folder that holds agents by its path from the root",
        ),
        (
            {"support": 1, "support/billing": 2},
            "engine: workers names 'support/billing' inside 'support', which it names too; keep one",
        ),
        (
            {"support/billing": 2, "support": 1},
            "engine: workers names 'support/billing' inside 'support', which it names too; keep one",
        ),
        (
            {"support": 0},
            "engine: workers gives 'support' 0 processes; give each folder 1 or more",
        ),
    ],
)
def test_workers_names_folders_of_the_tree_none_inside_another(
    tmp_path: Path, workers: dict[str, int], said: str
) -> None:
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    with pytest.raises(ConfigurationError) as caught:
        Engine(tmp_path, models={}, workers=workers)
    assert str(caught.value) == said


@pytest.mark.parametrize("name", ["shared/tools", "shared", "."])
def test_workers_refuses_a_folder_that_holds_no_agent(tmp_path: Path, name: str) -> None:
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    write(tmp_path / "shared/tools/x.txt", "x")
    with pytest.raises(ConfigurationError) as caught:
        Engine(tmp_path, models={}, workers={name: 2})
    assert str(caught.value) == (
        f"engine: workers names {name!r}, which holds no agent of the tree; name a folder that holds agents by its path from the root"
    )


@pytest.mark.parametrize("name", ["a", "a/b", "a/b/c"])
def test_workers_accepts_a_folder_that_holds_agents(tmp_path: Path, name: str) -> None:
    write(tmp_path / "a/b/c/AGENT.md", agent_md("c"))
    Engine(tmp_path, models={}, workers={name: 2})


def test_an_engine_walks_the_tree_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    walked: list[Path] = []
    real = tree._tree_folders

    def walk(root: Path) -> object:
        walked.append(root)
        return real(root)

    monkeypatch.setattr(tree, "_tree_folders", walk)
    Engine(tmp_path, models={}, workers={"a": 1})
    assert len(walked) == 1


def ended(pid: int) -> bool:
    for _ in range(40):
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_an_engine_called_from_one_event_loop_after_another_follows_each(tmp_path: Path) -> None:
    tool(shop(tmp_path) / "shop/tools", "pid", body=PID)
    engine = Engine(tmp_path, models={})
    surface = EngineTools(engine, agent="shop", scope="acme")

    async def call() -> int:
        async with asyncio.timeout(10):
            return pid(await surface.dispatch("pid", {}))

    async def close() -> None:
        async with asyncio.timeout(10):
            await engine.close()

    first = asyncio.run(call())
    second = asyncio.run(call())
    assert second != first and ended(first)
    asyncio.run(close())
    assert ended(second)


async def test_an_engine_starts_its_workers_at_the_first_call_and_close_stops_them(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "loaded"
    body = f"import os\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\ndef run(args, call):\n    return os.getpid()\n"
    tool(shop(tmp_path) / "shop/tools", "pid", body=body)
    engine = Engine(tmp_path, models={})
    await asyncio.sleep(0.3)
    assert not marker.exists()
    first = pid(await EngineTools(engine, agent="shop", scope="acme").dispatch("pid", {}))
    assert marker.exists() and alive(first)
    await engine.close()
    assert await gone(first)
    with pytest.raises(InputError, match="^engine: this engine was closed; build a new Engine$"):
        EngineTools(engine, agent="shop", scope="acme")
    async with Engine(tmp_path, models={}) as again:
        second = pid(await EngineTools(again, agent="shop", scope="acme").dispatch("pid", {}))
    assert await gone(second)


async def test_a_closed_engine_refuses_every_use_and_records_nothing(
    tmp_path: Path, storage: Storage
) -> None:
    tool(shop(tmp_path, extra="description: A shop\n") / "shop/tools", "pid", body=PID)
    write(tmp_path / "judge/AGENT.md", agent_md("judge", kind="decision", model="judge"))
    write(tmp_path / "judge/questions/fine.json", '{"type": "check", "instructions": "Fine?"}')
    refinery = Refinery(storage)
    llm = Scripted(calling(("pid", {})), text_reply("done"))
    engine = Engine(tmp_path, models={"strong": llm, "judge": Answering()}, refinery=refinery)
    asking = engine.tool("shop", scope="acme")
    recorded = await generated(engine, "shop", "acme", "x")
    first = pid(json.loads(str(recorded.steps[0].result)))
    await engine.close()
    await engine.close()
    assert await gone(first)
    said = "engine: this engine was closed; build a new Engine"
    with pytest.raises(InputError) as raised:
        await engine.run("shop", "acme", "x")
    assert str(raised.value) == said
    with pytest.raises(InputError, match=f"^{said}$"):
        await engine.run("judge", "acme", "x")
    with pytest.raises(InputError, match=f"^{said}$"):
        await asking.dispatch("shop", {"input": "x"})
    with pytest.raises(InputError, match=f"^{said}$"):
        EngineTools(engine, agent="shop", scope="acme")
    pairs = [pair async for pair in refinery.replay("acme", engine, agent="shop")]
    assert [(pair.candidate, pair.error) for pair in pairs] == [(None, f"InputError: {said}")]
    labeled = await refinery.label("acme", engine, judge="judge")
    assert labeled.judged == 0 and labeled.errors == {recorded.run_id: f"InputError: {said}"}
    assert [row.run_id for row in await storage.runs("acme")] == [recorded.run_id]
    assert len(llm.calls) == 2


async def test_a_run_in_flight_when_its_engine_closes_is_refused_the_runs_it_would_start(
    tmp_path: Path,
) -> None:
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="agents: [aide, judge]\n"))
    write(tmp_path / "aide/AGENT.md", agent_md("aide", extra="description: Helps\n"))
    write(
        tmp_path / "judge/AGENT.md",
        agent_md("judge", kind="decision", model="judge", extra="description: Judges\n"),
    )
    write(tmp_path / "judge/questions/fine.json", '{"type": "check", "instructions": "Fine?"}')
    llm = Scripted(
        calling(("close", {})),
        calling(
            ("Delegate", {"agent": "aide", "input": "x"}),
            ("Delegate", {"agent": "judge", "input": "x"}),
        ),
        text_reply("done"),
    )
    judging = Answering()
    engine = Engine(tmp_path, models={"strong": llm, "judge": judging})

    async def close(args: dict[str, Any]) -> str:
        await engine.close()
        return "closed"

    run = await generated(
        engine, "lead", "acme", "x", tools=Tools([Tool("close", "Close", body=close)])
    )
    said = "InputError: engine: this engine was closed; build a new Engine"
    assert run.reply == "done" and run.children == []
    assert [(step.result, step.is_error) for step in run.steps] == [
        ("closed", False),
        (said, True),
        (said, True),
    ]
    assert len(llm.calls) == 3 and judging.calls == []


@pytest.mark.parametrize("target", ["aide", "checker"])
async def test_a_route_after_its_engine_closes_fails_the_decision_run(
    tmp_path: Path, target: str
) -> None:
    write(
        tmp_path / "triage/AGENT.md",
        agent_md(
            "triage",
            kind="decision",
            model="judge",
            extra=f"route: {{by: go, to: {{next: {target}}}}}\n",
        ),
    )
    write(
        tmp_path / "triage/questions/go.json",
        '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep", "next": "Pass"}}',
    )
    write(tmp_path / "aide/AGENT.md", agent_md("aide"))
    write(tmp_path / "checker/AGENT.md", agent_md("checker", kind="decision", model="judge"))
    write(tmp_path / "checker/questions/fine.json", '{"type": "check", "instructions": "Fine?"}')
    engine: Engine

    class Closing(Answering):
        async def decide(self, state: Any, questions: Any) -> Any:
            await engine.close()
            return await super().decide(state, questions)

    llm = Scripted()
    closing = Closing()
    engine = Engine(tmp_path, models={"strong": llm, "judge": closing})
    with pytest.raises(InputError) as raised:
        await engine.run("triage", "acme", "x")
    assert str(raised.value) == "engine: this engine was closed; build a new Engine"
    assert llm.calls == [] and [list(questions) for _, questions in closing.calls] == [["go"]]


async def test_an_edit_to_a_script_after_the_engine_loads_runs_in_the_next_engine_only(
    tmp_path: Path,
) -> None:
    script = tmp_path / "shop/tools/v.py"
    tool(shop(tmp_path) / "shop/tools", "v", body="def run(args, call):\n    return 'loaded'\n")
    async with Engine(tmp_path, models={}) as engine:
        fingerprint = engine._spec("shop").fingerprint
        script.write_text("def run(args, call):\n    return 'edited'\n", encoding="utf-8")
        assert await EngineTools(engine, agent="shop", scope="acme").dispatch("v", {}) == "loaded"
    async with Engine(tmp_path, models={}) as again:
        assert await EngineTools(again, agent="shop", scope="acme").dispatch("v", {}) == "edited"
        assert again._spec("shop").fingerprint != fingerprint


async def test_an_edit_to_a_helper_after_the_engine_loads_does_not_change_what_runs(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "shop/tools/_client.py"
    write(helper, "NAME = 'loaded'\n")
    body = "import _client\ndef run(args, call):\n    return _client.NAME\n"
    tool(shop(tmp_path) / "shop/tools", "v", body=body)
    async with Engine(tmp_path, models={}) as engine:
        write(helper, "NAME = 'edited'\n")
        assert await EngineTools(engine, agent="shop", scope="acme").dispatch("v", {}) == "loaded"


async def test_a_worker_started_again_after_a_crash_runs_the_source_the_tree_read(
    tmp_path: Path,
) -> None:
    script = tmp_path / "shop/tools/v.py"
    body = "import os\ndef run(args, call):\n    if args.get('die'):\n        os._exit(3)\n    return 'loaded'\n"
    tool(shop(tmp_path) / "shop/tools", "v", body=body)
    async with Engine(tmp_path, models={}) as engine:
        surface = EngineTools(engine, agent="shop", scope="acme")
        assert await surface.dispatch("v", {}) == "loaded"
        script.write_text("def run(args, call):\n    return 'edited'\n", encoding="utf-8")
        with pytest.raises(WorkerError, match="exited with code 3"):
            await surface.dispatch("v", {"die": True})
        assert await surface.dispatch("v", {}) == "loaded"


def test_engine_takes_no_tools(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="tools"):
        Engine(shop(tmp_path), models={}, tools=[])  # type: ignore[call-arg]


async def test_each_tenant_s_run_hands_its_own_scope_to_the_tool(tmp_path: Path) -> None:
    body = "import time\ndef run(args, call):\n    time.sleep(0.2)\n    return call['scope']\n"
    tool(shop(tmp_path) / "shop/tools", "whose", body=body)
    llm = Scripted(
        calling(("whose", {})), calling(("whose", {})), text_reply("done"), text_reply("done")
    )
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        runs = await asyncio.gather(
            generated(engine, "shop", "acme", "x"), generated(engine, "shop", "globex", "x")
        )
    assert {run.scope: run.steps[0].result for run in runs} == {
        "acme": "acme",
        "globex": "globex",
    }


async def test_engine_tools_holds_exactly_the_assembled_set(tmp_path: Path) -> None:
    write(tmp_path / "shop/AGENT.md", agent_md("shop", extra="agents: [aide]\n"))
    write(tmp_path / "aide/AGENT.md", agent_md("aide", extra="description: Helps\n"))
    tool(tmp_path / "tools", "notify", body="def run(args, call):\n    return 'notified'\n")
    tool(tmp_path / "shop/tools", "lookup_order", body="def run(args, call):\n    return args\n")
    ping = Tools([Tool("ping", "Ping", body=pong)])
    async with Engine(tmp_path, models={}) as engine:
        surface = EngineTools(engine, agent="shop", scope="acme", tools=ping)
        assert surface.names == [
            "Read",
            "Grep",
            "Glob",
            "LS",
            "Load",
            "lookup_order",
            "notify",
            "Delegate",
            "ping",
        ]
        assert await surface.dispatch("lookup_order", {"order_id": "A-1"}) == {"order_id": "A-1"}
        assert await surface.dispatch("notify", {}) == "notified"
        assert await surface.dispatch("ping", {}) == "pong"


async def test_agent_info_lists_the_folder_tools_an_agent_gets(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "notify")
    write(
        tmp_path / "support/billing/AGENT.md",
        agent_md("billing", extra="tools: [lookup_order, refund]\n"),
    )
    tool(tmp_path / "support/billing/tools", "refund")
    tool(tmp_path / "support/tools", "lookup_order")
    write(tmp_path / "support/AGENT.md", agent_md("support", extra="tools: [notify, Read]\n"))
    async with Engine(tmp_path, models={}) as engine:
        assert engine.agents["billing"].tools == ["lookup_order", "refund"]
    assert engine.agents["support"].tools == ["notify"]


async def test_a_relative_root_hands_the_workers_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "def run(args, call):\n    return __file__\n"
    tool(shop(tmp_path / "agents") / "shop/tools", "where", body=body)
    monkeypatch.chdir(tmp_path)
    where = load_tree(Path("agents"))["shop"].folder_tools["where"]
    assert where.script.is_absolute() and where.folder.is_absolute()
    assert where.script.resolve() == (tmp_path / "agents/shop/tools/where.py").resolve()
    async with Engine("agents", models={}) as engine:
        monkeypatch.chdir(tmp_path / "agents/shop")
        found = await EngineTools(engine, agent="shop", scope="acme").dispatch("where", {})
    assert found == str((tmp_path / "agents/shop/tools/where.py").resolve())


SHOWN = "import _read\ndef run(args, call):\n    return [_read.WHO, _read.__file__]\n"


async def test_a_script_runs_with_the_helpers_of_the_tools_folders_above_it(
    tmp_path: Path,
) -> None:
    write(tmp_path / "tools/_read.py", "import _base\nWHO = 'above, ' + _base.NAME\n")
    write(tmp_path / "tools/_base.py", "NAME = 'with its own helper'\n")
    tool(tmp_path / "tools", "outer", body=SHOWN)
    write(tmp_path / "alpha/AGENT.md", agent_md("alpha"))
    tool(tmp_path / "alpha/tools", "inner", body=SHOWN)
    write(tmp_path / "beta/AGENT.md", agent_md("beta"))
    write(tmp_path / "beta/tools/_read.py", "WHO = 'near'\n")
    tool(tmp_path / "beta/tools", "local", body=SHOWN)
    above = ["above, with its own helper", str(tmp_path / "tools/_read.py")]
    async with Engine(tmp_path, models={}) as engine:
        alpha = EngineTools(engine, agent="alpha", scope="acme")
        beta = EngineTools(engine, agent="beta", scope="acme")
        shown: list[object] = [
            await alpha.dispatch("inner", {}),
            await alpha.dispatch("outer", {}),
            await beta.dispatch("local", {}),
            await beta.dispatch("outer", {}),
        ]
    assert shown == [above, above, ["near", str(tmp_path / "beta/tools/_read.py")], above]
