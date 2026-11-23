from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Answering, Crew, Scripted, background, crew, team, text_reply
from .test_engine_nesting import _agents, _delegate, _router
from .trees import add_agent, provided

CONTEXT = "Acme is enterprise."


def _child(run: Any) -> Generation:
    child = run.children[0]
    assert isinstance(child, Generation)
    return child


async def test_a_delegated_run_reads_its_starter_s_context_by_default(tmp_path: Path) -> None:
    _agents(tmp_path, a=["b"], b=[])
    llm = Scripted(_delegate("b", "go"), text_reply("b done"), text_reply("a done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("a", "acme", "start", context=CONTEXT)
    child = _child(run)
    assert child.context == CONTEXT and CONTEXT in child.system


async def test_an_own_agent_started_by_delegate_reads_no_context(tmp_path: Path) -> None:
    _agents(tmp_path, a=["b"], b=[])
    add_agent(
        tmp_path,
        "b",
        {"kind": "generation", "model": "strong", "instructions": "b.md", "context": "own"},
        files={"b.md": "hi\n"},
    )
    llm = Scripted(_delegate("b", "go"), text_reply("b done"), text_reply("a done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("a", "acme", "start", context=CONTEXT)
    child = _child(run)
    assert child.context is None and CONTEXT not in child.system
    assert isinstance(run, Generation)
    assert run.context == CONTEXT and CONTEXT in run.system


async def test_an_own_helper_in_the_background_reads_no_context(tmp_path: Path) -> None:
    model = Crew(
        [background("part"), text_reply("started"), text_reply("done")],
        {"part": [text_reply("ok")]},
    )
    root = crew(tmp_path, worker={"context": "own"}, background=True)
    run = await team(root, model).run("lead", "acme", "go", context=CONTEXT)
    helper = _child(run)
    assert helper.context is None and CONTEXT not in helper.system
    assert isinstance(run, Generation)
    assert run.context == CONTEXT


async def test_an_own_agent_reads_its_own_context_at_the_top(tmp_path: Path) -> None:
    _agents(tmp_path, b=[])
    add_agent(
        tmp_path,
        "b",
        {"kind": "generation", "model": "strong", "instructions": "b.md", "context": "own"},
        files={"b.md": "hi\n"},
    )
    llm = Scripted(text_reply("b done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("b", "acme", "start", context="X")
    assert isinstance(run, Generation)
    assert run.context == "X" and "X" in run.system


async def test_a_routed_run_reads_the_router_s_context_whatever_its_entry_says(
    tmp_path: Path,
) -> None:
    _agents(tmp_path, g=[])
    add_agent(
        tmp_path,
        "g",
        {"kind": "generation", "model": "strong", "instructions": "g.md", "context": "own"},
        files={"g.md": "hi\n"},
    )
    _router(tmp_path, "router", "g")
    llm = Scripted(text_reply("routed"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": Answering()}))
    run = await engine.run("router", "acme", "start", context="On call: Ana.")
    assert isinstance(run, Decision)
    assert isinstance(run.routed_to, Generation)
    assert run.routed_to is not None and run.routed_to.context == "On call: Ana."


async def test_inheritance_is_from_the_starter_not_the_top(tmp_path: Path) -> None:
    _agents(tmp_path, a=["b"], b=["c"], c=[])
    add_agent(
        tmp_path,
        "b",
        {
            "kind": "generation",
            "model": "strong",
            "instructions": "b.md",
            "context": "own",
            "orchestration": {"delegate_to": ["c"]},
        },
        files={"b.md": "hi\n"},
    )
    llm = Scripted(
        _delegate("b", "go"),
        _delegate("c", "go"),
        text_reply("c done"),
        text_reply("b done"),
        text_reply("a done"),
    )
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("a", "acme", "start", context=CONTEXT)
    b = _child(run)
    c = _child(b)
    assert b.context is None and c.context is None


async def test_an_own_decision_agent_routes_on_without_context(tmp_path: Path) -> None:
    _agents(tmp_path, a=["d"], g=[])
    add_agent(
        tmp_path,
        "d",
        {
            "kind": "decision",
            "model": "judge",
            "instructions": "d.md",
            "context": "own",
            "questions": ["d/*.json"],
            "orchestration": {"route": {"by": "go", "to": {"next": "g"}}},
        },
        files={
            "d.md": "",
            "d/go.json": '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep it", "next": "Pass it on"}}',
        },
    )
    llm = Scripted(_delegate("d", "go"), text_reply("g done"), text_reply("a done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": Answering()}))
    run = await engine.run("a", "acme", "start", context=CONTEXT)
    assert isinstance(run, Generation)
    d = run.children[0]
    assert isinstance(d, Decision) and d.routed_to is not None
    assert isinstance(d.routed_to, Generation)
    assert d.routed_to.context is None


async def test_a_delegate_through_engine_tools_starts_runs_without_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _agents(tmp_path, a=["b"], b=[])
    llm = Scripted(text_reply("b done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": None}))
    held: list[str | None] = []
    original = engine._run

    async def spying(spec: Any, scope: Any, input: Any, options: Any, *rest: Any, **kw: Any) -> Any:
        held.append(options.context)
        return await original(spec, scope, input, options, *rest, **kw)

    monkeypatch.setattr(engine, "_run", spying)
    tools = EngineTools(engine, agent="a", scope="acme")
    assert await tools.dispatch("Delegate", {"agent": "b", "input": "go"}) == "b done"
    assert held == [None]


async def test_the_delegate_description_marks_an_own_agent(tmp_path: Path) -> None:
    _agents(tmp_path, lead=["b", "c", "d", "e"], b=[], c=[])
    _router(tmp_path, "d", "c")
    _router(tmp_path, "e", "c")
    add_agent(
        tmp_path,
        "b",
        {
            "kind": "generation",
            "model": "strong",
            "instructions": "b.md",
            "description": "Does b",
            "context": "own",
        },
        files={"b.md": "hi\n"},
    )
    add_agent(
        tmp_path,
        "c",
        {
            "kind": "generation",
            "model": "strong",
            "instructions": "c.md",
            "description": "Does c",
            "context": "inherit",
        },
        files={"c.md": "hi\n"},
    )
    for name in ("d", "e"):
        entry = {
            "kind": "decision",
            "model": "judge",
            "instructions": f"{name}.md",
            "description": f"Judges {name}",
            "questions": [f"{name}/*.json"],
            "orchestration": {"route": {"by": "go", "to": {"next": "c"}}},
        }
        if name == "d":
            entry["context"] = "own"
        add_agent(
            tmp_path,
            name,
            entry,
            files={
                f"{name}.md": "",
                f"{name}/go.json": '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep it", "next": "Pass it on"}}',
            },
        )
    llm = Scripted(text_reply("done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("lead", "acme", "go")
    assert isinstance(run, Generation)
    [tool] = run.tools
    lines = tool.description.splitlines()
    assert "- b (sees only its input, not your context): Does b" in lines
    assert "- c: Does c" in lines
    assert "- d (decision agent) (sees only its input, not your context): Judges d" in lines
    assert "- e (decision agent): Judges e" in lines


async def test_writing_context_inherit_keeps_every_fingerprint(tmp_path: Path) -> None:
    async def fingerprints(root: Path, held: str | None) -> tuple[str, str, str]:
        _agents(root, lead=["b"], b=[])
        if held is not None:
            add_agent(
                root,
                "b",
                {"kind": "generation", "model": "strong", "instructions": "b.md", "context": held},
                files={"b.md": "hi\n"},
            )
        llm = Scripted(_delegate("b", "go"), text_reply("b done"), text_reply("done"))
        engine = Engine(root, providers=provided({"strong": llm, "judge": None}))
        run = await engine.run("lead", "acme", "go")
        return (
            run.fingerprint.agent,
            run.fingerprint.tools,
            _child(run).fingerprint.agent,
        )

    absent = await fingerprints(tmp_path / "absent", None)
    inherit = await fingerprints(tmp_path / "inherit", "inherit")
    own = await fingerprints(tmp_path / "own", "own")
    assert absent == inherit
    assert own[0] == absent[0] and own[1] != absent[1] and own[2] != absent[2]
