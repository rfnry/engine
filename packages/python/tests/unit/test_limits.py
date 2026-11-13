import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import RunawayError
from rfnry_engine.models import Message, Reply, ToolCall, ToolSpec
from rfnry_engine.records import Generation, Run
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted, delegating, text_reply
from .test_engine_bridge import script, shell
from .test_engine_workspace import use
from .trees import picked_by, provided, runtime, write_folder

TOO_DEEP = "DelegationDepthError: tools/delegate: depth 2 exceeded at a -> b -> c"
TOO_LATE = "TimeoutError: engine: slow timed out after 0.05s"
TOO_MANY = "engine: run exceeded 2 model calls"
FAN_OUT = """
import threading
threads = [threading.Thread(target=call, args=("probe",)) for _ in range(5)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""


class Gauge:
    def __init__(self) -> None:
        self.now = 0
        self.peak = 0

    async def hold(self) -> None:
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.1)
        self.now -= 1


class Relaying:
    def __init__(self, to: str, together: asyncio.Barrier | None = None) -> None:
        self.to = to
        self.together = together
        self.heard: list[Any] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if messages[-1].role == "tool":
            self.heard.append(messages[-1].content)
            return text_reply("relayed")
        if self.together is not None:
            await self.together.wait()
        return delegating(self.to)


class Measured:
    def __init__(self, gauge: Gauge) -> None:
        self.gauge = gauge

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        await self.gauge.hold()
        return text_reply("child done")


def agent(**sections: Any) -> dict[str, Any]:
    return {
        "kind": "generation",
        "model": "m",
        "instructions": "a.md",
        "description": "Helps",
        **sections,
    }


def folder(tmp_path: Path, agents: Mapping[str, Mapping[str, Any]]) -> Path:
    return write_folder(
        tmp_path, agents, runtime_tools=runtime(picked_by(agents)), files={"a.md": "Work.\n"}
    )


def calling(*names: str) -> Reply:
    calls = [ToolCall(id=f"c{n}", name=name, args={}) for n, name in enumerate(names)]
    return Reply(tool_calls=calls, stop_reason="tool_use")


def fanning_out(*agents: str) -> Reply:
    calls = [
        ToolCall(id=f"d{n}", name="Delegate", args={"agent": name, "input": "go"})
        for n, name in enumerate(agents)
    ]
    return Reply(tool_calls=calls, stop_reason="tool_use")


def tool(body: Callable[[dict[str, Any]], Awaitable[Any]], name: str = "probe") -> Tools:
    return Tools([Tool(name, "Does a thing", body=body)])


async def done(args: dict[str, Any]) -> str:
    return "done"


PROBE = tool(done)


def worked(run: Run) -> Generation:
    assert isinstance(run, Generation)
    assert [step.result for step in run.steps if step.is_error] == []
    return run


async def test_max_model_calls_counts_model_calls_of_one_run(tmp_path: Path) -> None:
    written = folder(tmp_path, {"a": agent(max_model_calls=2, tools={"allow": ["probe"]})})
    model = Scripted(calling("probe", "probe", "probe"), calling("probe"), text_reply("never"))
    with pytest.raises(RunawayError, match=f"^{TOO_MANY}$"):
        await Engine(written, providers=provided({"m": model})).run("a", "acme", "go", tools=PROBE)
    assert len(model.calls) == 2
    ends_in_time = Scripted(calling("probe"), text_reply("done"))
    run = await Engine(written, providers=provided({"m": ends_in_time})).run(
        "a", "acme", "go", tools=PROBE
    )
    assert worked(run).reply == "done"


async def test_a_delegated_run_counts_its_own(tmp_path: Path) -> None:
    (tmp_path / "enough").mkdir()
    (tmp_path / "over").mkdir()
    enough = folder(
        tmp_path / "enough",
        {
            "a": agent(max_model_calls=2, orchestration={"delegate_to": ["b"]}),
            "b": agent(max_model_calls=2, tools={"allow": ["probe"]}),
        },
    )
    model = Scripted(delegating("b"), calling("probe"), text_reply("b done"), text_reply("a done"))
    run = worked(
        await Engine(enough, providers=provided({"m": model})).run("a", "acme", "go", tools=PROBE)
    )
    assert run.reply == "a done"
    assert worked(run.children[0]).reply == "b done"
    assert len(model.calls) == 4
    over = folder(
        tmp_path / "over",
        {
            "a": agent(max_model_calls=2, orchestration={"delegate_to": ["b"]}),
            "b": agent(max_model_calls=1, tools={"allow": ["probe"]}),
        },
    )
    model = Scripted(delegating("b"), calling("probe"), text_reply("a done"))
    refused = await Engine(over, providers=provided({"m": model})).run(
        "a", "acme", "go", tools=PROBE
    )
    assert isinstance(refused, Generation) and refused.reply == "a done"
    assert refused.steps[0].result == "RunawayError: engine: run exceeded 1 model calls"


async def model_calls_refused(tmp_path: Path, written: Any, ceiling: Any) -> str:
    root = folder(tmp_path, {"a": agent(max_model_calls=written, tools={"allow": ["probe"]})})
    model = Scripted(*[calling("probe") for _ in range(9)])
    engine = Engine(root, providers=provided({"m": model}), max_model_calls=ceiling)
    with pytest.raises(RunawayError) as raised:
        await engine.run("a", "acme", "go", tools=PROBE)
    return str(raised.value)


async def depth_refused(tmp_path: Path, written: Any, ceiling: Any) -> str:
    root = folder(
        tmp_path,
        {
            "a": agent(orchestration={"delegate_to": ["b"], "max_depth": written}),
            "b": agent(orchestration={"delegate_to": ["c"]}),
            "c": agent(),
        },
    )
    model = Scripted(delegating("b"), delegating("c"), text_reply("b done"), text_reply("a done"))
    run = await Engine(root, providers=provided({"m": model}), max_depth=ceiling).run(
        "a", "acme", "go"
    )
    assert isinstance(run, Generation) and isinstance(run.children[0], Generation)
    return str(run.children[0].steps[0].result)


async def deadline_passed(tmp_path: Path, written: Any, ceiling: Any) -> str:
    async def slow(args: dict[str, Any]) -> str:
        await asyncio.sleep(2)
        return "done"

    root = folder(tmp_path, {"a": agent(tools={"allow": ["slow"], "timeout": written})})
    model = Scripted(calling("slow"), text_reply("done"))
    run = await Engine(root, providers=provided({"m": model}), tool_timeout=ceiling).run(
        "a", "acme", "go", tools=tool(slow, "slow")
    )
    assert isinstance(run, Generation)
    return str(run.steps[0].result)


@pytest.mark.parametrize(
    ("refusal", "written", "ceiling", "expected"),
    [
        pytest.param(model_calls_refused, 2, 5, TOO_MANY, id="calls-lowered"),
        pytest.param(model_calls_refused, 5, 2, TOO_MANY, id="calls-not-raised"),
        pytest.param(depth_refused, 1, 5, TOO_DEEP, id="depth-lowered"),
        pytest.param(depth_refused, 4, 2, TOO_DEEP, id="depth-not-raised"),
        pytest.param(deadline_passed, 0.05, 0.3, TOO_LATE, id="timeout-lowered"),
        pytest.param(deadline_passed, 0.3, 0.05, TOO_LATE, id="timeout-not-raised"),
        pytest.param(deadline_passed, 0.05, None, TOO_LATE, id="timeout-no-ceiling"),
    ],
)
async def test_the_folder_lowers_and_cannot_raise(
    tmp_path: Path,
    refusal: Callable[[Path, Any, Any], Awaitable[str]],
    written: Any,
    ceiling: Any,
    expected: str,
) -> None:
    assert await refusal(tmp_path, written, ceiling) == expected


@pytest.mark.parametrize(
    ("first", "chain"),
    [("a", "depth 2 exceeded at a -> b -> c"), ("x", "depth 3 exceeded at x -> a -> b -> c")],
)
async def test_max_depth_counts_levels_below_the_agent(
    tmp_path: Path, first: str, chain: str
) -> None:
    root = folder(
        tmp_path,
        {
            "x": agent(orchestration={"delegate_to": ["a"]}),
            "a": agent(orchestration={"delegate_to": ["b"], "max_depth": 1}),
            "b": agent(orchestration={"delegate_to": ["c"], "max_depth": 3}),
            "c": agent(),
        },
    )
    replies = [delegating("b"), delegating("c"), text_reply("b done"), text_reply("a done")]
    if first == "x":
        replies = [delegating("a"), *replies, text_reply("x done")]
    model = Scripted(*replies)
    run: Run = await Engine(root, providers=provided({"m": model})).run(first, "acme", "go")
    assert isinstance(run, Generation)
    if first == "x":
        run = worked(run).children[0]
    b = worked(run).children[0]
    assert isinstance(b, Generation) and b.agent == "b" and b.children == []
    assert (
        b.steps[0].is_error
        and b.steps[0].result == f"DelegationDepthError: tools/delegate: {chain}"
    )


async def test_the_limits_of_two_delegates_at_once_stay_apart(tmp_path: Path) -> None:
    together = asyncio.Barrier(2)
    leaf = Scripted(text_reply("c done"))
    root = folder(
        tmp_path,
        {
            "p": agent(model="p", orchestration={"delegate_to": ["tight", "wide"]}),
            "tight": agent(model="tight", orchestration={"delegate_to": ["b"], "max_depth": 1}),
            "wide": agent(model="wide", orchestration={"delegate_to": ["b"], "max_depth": 3}),
            "b": agent(model="b", orchestration={"delegate_to": ["c"]}),
            "c": agent(model="c"),
        },
    )
    async with asyncio.timeout(5):
        run = await Engine(
            root,
            providers=provided(
                {
                    "p": Scripted(fanning_out("tight", "wide"), text_reply("done")),
                    "tight": Relaying("b", together),
                    "wide": Relaying("b", together),
                    "b": Relaying("c"),
                    "c": leaf,
                }
            ),
        ).run("p", "acme", "go")
    tight, wide = worked(run).children
    assert isinstance(tight, Generation) and isinstance(wide, Generation)
    below_tight, below_wide = tight.children[0], wide.children[0]
    assert isinstance(below_tight, Generation) and isinstance(below_wide, Generation)
    assert below_tight.steps[0].result == (
        "DelegationDepthError: tools/delegate: depth 3 exceeded at p -> tight -> b -> c"
    )
    assert below_wide.steps[0].result == "c done"


@pytest.mark.parametrize(
    ("written", "ceiling", "heard"),
    [
        pytest.param(1, 5, TOO_DEEP, id="the-folder-limit-follows"),
        pytest.param(None, 2, "c done", id="the-ceiling-stays-with-its-engine"),
    ],
)
async def test_limits_across_engine_tool_of_another_engine(
    tmp_path: Path, written: int | None, ceiling: int, heard: str
) -> None:
    (tmp_path / "outer").mkdir()
    (tmp_path / "inner").mkdir()
    limit = {} if written is None else {"orchestration": {"max_depth": written}}
    outer = folder(tmp_path / "outer", {"a": agent(tools={"allow": ["b"]}, **limit)})
    inner = folder(
        tmp_path / "inner",
        {
            "b": agent(model="b", orchestration={"delegate_to": ["c"]}),
            "c": agent(model="c"),
        },
    )
    relay = Relaying("c")
    consult = Reply(
        tool_calls=[ToolCall(id="t", name="b", args={"input": "go"})], stop_reason="tool_use"
    )
    other = Engine(inner, providers=provided({"b": relay, "c": Scripted(text_reply("c done"))}))
    model = Scripted(consult, text_reply("done"))
    run = await Engine(outer, providers=provided({"m": model}), max_depth=ceiling).run(
        "a", "acme", "go", tools=other.tool("b", scope="acme")
    )
    assert worked(run).reply == "done"
    assert relay.heard == [heard]


async def peak_of_tool_calls(tmp_path: Path, section: Mapping[str, Any]) -> int:
    gauge = Gauge()

    async def measured(args: dict[str, Any]) -> str:
        await gauge.hold()
        return "done"

    root = folder(tmp_path, {"a": agent(tools={"allow": ["probe"], **section})})
    model = Scripted(calling(*["probe"] * 5), text_reply("done"))
    worked(
        await Engine(root, providers=provided({"m": model})).run(
            "a", "acme", "go", tools=tool(measured)
        )
    )
    return gauge.peak


async def peak_of_delegate_calls(tmp_path: Path, section: Mapping[str, Any]) -> int:
    gauge = Gauge()
    root = folder(
        tmp_path,
        {
            "a": agent(orchestration={"delegate_to": ["b"], **section}),
            "b": agent(model="child"),
        },
    )
    model = Scripted(fanning_out(*["b"] * 4), text_reply("done"))
    run = await Engine(root, providers=provided({"m": model, "child": Measured(gauge)})).run(
        "a", "acme", "go"
    )
    assert len(worked(run).children) == 4
    return gauge.peak


async def test_tools_max_parallel_caps_a_runs_tool_calls(tmp_path: Path) -> None:
    assert await peak_of_tool_calls(tmp_path, {"max_parallel": 2}) == 2


async def test_orchestration_max_parallel_caps_delegate_calls(tmp_path: Path) -> None:
    assert await peak_of_delegate_calls(tmp_path, {"max_parallel": 2}) == 2


async def test_each_run_has_its_own_slots(tmp_path: Path) -> None:
    gauge = Gauge()

    async def measured(args: dict[str, Any]) -> str:
        await gauge.hold()
        return "done"

    root = folder(tmp_path, {"a": agent(tools={"allow": ["probe"], "max_parallel": 1})})
    model = Scripted(calling("probe"), calling("probe"), text_reply("done"), text_reply("done"))
    engine = Engine(root, providers=provided({"m": model}))
    runs = await asyncio.gather(
        *[engine.run("a", "acme", "go", tools=tool(measured)) for _ in range(2)]
    )
    assert [worked(run).reply for run in runs] == ["done", "done"]
    assert gauge.peak == 2


async def test_a_delegate_call_takes_no_tool_slot(tmp_path: Path) -> None:
    reached = asyncio.Event()

    async def opens(args: dict[str, Any]) -> str:
        reached.set()
        return "done"

    class Waiting:
        async def generate(
            self, system: str, messages: list[Message], tools: list[ToolSpec]
        ) -> Reply:
            await reached.wait()
            return text_reply("child done")

    root = folder(
        tmp_path,
        {
            "a": agent(
                tools={"allow": ["probe"], "max_parallel": 1},
                orchestration={"delegate_to": ["b"]},
            ),
            "b": agent(model="child"),
        },
    )
    together = Reply(
        tool_calls=[
            ToolCall(id="d", name="Delegate", args={"agent": "b", "input": "go"}),
            ToolCall(id="p", name="probe", args={}),
        ],
        stop_reason="tool_use",
    )
    model = Scripted(together, text_reply("done"))
    async with asyncio.timeout(2):
        run = await Engine(root, providers=provided({"m": model, "child": Waiting()})).run(
            "a", "acme", "go", tools=tool(opens)
        )
    assert worked(run).reply == "done"


async def test_a_call_waiting_on_approve_holds_no_slot(tmp_path: Path) -> None:
    other_ran = asyncio.Event()

    async def approve(name: str, call: ToolCall) -> bool:
        if call.id == "c0":
            await other_ran.wait()
        return True

    async def runs(args: dict[str, Any]) -> str:
        other_ran.set()
        return "done"

    root = folder(tmp_path, {"a": agent(tools={"allow": ["probe"], "max_parallel": 1})})
    model = Scripted(calling("probe", "probe"), text_reply("done"))
    async with asyncio.timeout(2):
        run = await Engine(root, providers=provided({"m": model})).run(
            "a", "acme", "go", tools=tool(runs), approve=approve
        )
    assert worked(run).reply == "done"


async def test_a_call_waiting_for_a_slot_keeps_its_whole_deadline(tmp_path: Path) -> None:
    async def takes_a_while(args: dict[str, Any]) -> str:
        await asyncio.sleep(0.4)
        return "done"

    root = folder(
        tmp_path, {"a": agent(tools={"allow": ["probe"], "max_parallel": 1, "timeout": 0.6})}
    )
    model = Scripted(calling("probe", "probe"), text_reply("done"))
    run = await Engine(root, providers=provided({"m": model})).run(
        "a", "acme", "go", tools=tool(takes_a_while)
    )
    assert worked(run).reply == "done"


async def run_commands(
    tmp_path: Path,
    work: Path,
    reply: Reply,
    cap: int,
    tools: Tools = PROBE,
    bridge: bool = True,
    timeout: float = 5,
) -> Generation:
    root = folder(
        tmp_path,
        {
            "a": agent(
                workspace=True,
                tools={"allow": ["Shell", "probe"], "max_parallel": cap},
            )
        },
    )
    model = Scripted(reply, text_reply("done"))
    async with (
        Workspace(Local(work), scope="acme", timeout=timeout, bridge=bridge) as room,
        Engine(root, providers=provided({"m": model})) as engine,
    ):
        run = await engine.run("a", "acme", "go", workspace=room, tools=tools)
    assert isinstance(run, Generation)
    return run


async def test_a_call_through_the_bridge_finishes_under_a_cap_of_one(
    tmp_path: Path, short_root: Path
) -> None:
    script(short_root, "print(call('probe'))\n")
    run = await run_commands(tmp_path, short_root, use(shell()), cap=1)
    [bridged, command] = run.steps
    assert (bridged.tool, bridged.by, bridged.result) == ("probe", "1", "done")
    assert json.loads(str(command.result))["exit_code"] == 0


async def test_calls_through_the_bridge_stay_under_the_cap(
    tmp_path: Path, short_root: Path
) -> None:
    gauge = Gauge()

    async def measured(args: dict[str, Any]) -> str:
        await gauge.hold()
        return "done"

    script(short_root, FAN_OUT)
    run = await run_commands(tmp_path, short_root, use(shell()), cap=2, tools=tool(measured))
    *bridged, command = run.steps
    assert [(step.tool, step.is_error) for step in bridged] == [("probe", False)] * 5
    assert json.loads(str(command.result))["exit_code"] == 0
    assert gauge.peak == 2


@pytest.mark.parametrize(("bridge", "polls", "exit_code"), [(True, 200, 0), (False, 10, 7)])
async def test_a_shell_takes_a_slot_only_on_a_run_without_a_bridge(
    tmp_path: Path, short_root: Path, bridge: bool, polls: int, exit_code: int
) -> None:
    waits = f"i=0; while [ $i -lt {polls} ]; do [ -f second ] && exit 0; sleep 0.05; i=$((i+1)); done; exit 7"
    both = use(shell(waits, "1"), shell("touch second", "2"))
    run = await run_commands(tmp_path, short_root, both, cap=1, bridge=bridge, timeout=15)
    assert json.loads(str(run.steps[0].result))["exit_code"] == exit_code


@pytest.mark.parametrize(
    ("peak_of", "expected"), [(peak_of_tool_calls, 5), (peak_of_delegate_calls, 4)]
)
async def test_absent_caps_leave_calls_unbounded(
    tmp_path: Path,
    peak_of: Callable[[Path, Mapping[str, Any]], Awaitable[int]],
    expected: int,
) -> None:
    assert await peak_of(tmp_path, {}) == expected
