import asyncio
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Image, Message, Reply, Text, ToolCall, ToolSpec
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Answering, Scripted, delegating, text_reply
from .trees import write_folder

PNG = b"\x89PNG\r\n\x1a\ndiagram"
NOT_GIVEN = "which neither the engine nor the run was given"
TEAM = '{"type": "choice", "instructions": "Which?", "options": {"other": "Other", "writing": "Writing"}}'
UNFIT: list[Any] = [
    [("strong", 1)],
    {"": Scripted()},
    {" strong": Scripted()},
    {1: Scripted()},
    {"strong": object()},
]
LOAD = Reply(
    tool_calls=[ToolCall(id="l1", name="Load", args={"path": "diagram.png"})],
    stop_reason="tool_use",
)


class Pictures(Scripted):
    media_types = {"image/*"}


class Documents(Scripted):
    media_types = {"application/pdf"}


class Seeing(Answering):
    media_types = {"image/*"}


class Named(Scripted):
    def __init__(self, name: str, *replies: Reply) -> None:
        super().__init__(*replies)
        self.model = name


class NamedDecider(Answering):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.model = name


class Lockstep:
    def __init__(self, name: str, step: asyncio.Barrier, *replies: Reply) -> None:
        self.name, self.step = name, step
        self.replies = list(replies)
        self.calls = 0

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        await self.step.wait()
        self.calls += 1
        return self.replies.pop(0).model_copy(update={"model": self.name})


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return (
        f"---\nname: {name}\nkind: {kind}\nmodel: {model}\ndescription: Works\n{extra}---\nWork.\n"
    )


def pair(root: Path) -> Path:
    write(root / "lead/AGENT.md", agent_md("lead", extra="agents: [writer]\n"))
    write(root / "writer/AGENT.md", agent_md("writer", model="fast"))
    return root


def crew(root: Path) -> Path:
    write(root / "lead/AGENT.md", agent_md("lead", extra="agents: [triage, writer]\n"))
    write(
        root / "triage/AGENT.md",
        agent_md(
            "triage",
            kind="decision",
            model="judge",
            extra="route:\n  by: team\n  to: {writing: writer}\n",
        ),
    )
    write(root / "triage/questions/team.json", TEAM)
    write(root / "writer/AGENT.md", agent_md("writer", model="fast"))
    return root


def with_picture(root: Path) -> Path:
    write(root / "lead/AGENT.md", agent_md("lead", extra="agents: [writer]\n"))
    (root / "lead/diagram.png").write_bytes(PNG)
    write(root / "writer/AGENT.md", agent_md("writer", model="fast"))
    write(root / "grader/AGENT.md", agent_md("grader", kind="decision", model="judge"))
    write(root / "grader/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}')
    return root


def with_a_mode(root: Path) -> Path:
    return write_folder(
        root,
        {
            "a": {
                "kind": "generation",
                "model": "m",
                "instructions": "a.md",
                "modes": {"quick": {"model": "q"}},
            }
        },
        files={"a.md": "Work.\n"},
    )


async def test_run_models_win_over_the_engines_by_key(tmp_path: Path) -> None:
    kept, fast = Scripted(text_reply("the engine's")), Scripted(text_reply("drafted"))
    engine = Engine(pair(tmp_path), models={"strong": kept, "fast": fast})
    brought = Scripted(delegating("writer"), text_reply("the run's", model="run-1"))
    run = await engine.run("lead", "acme", "x", models={"strong": brought})
    assert isinstance(run, Generation)
    assert (run.reply, run.model) == ("the run's", "run-1")
    assert kept.calls == [] and len(brought.calls) == 2
    assert run.children[0].model is None and len(fast.calls) == 1
    afterwards = await engine.run("lead", "acme", "x")
    assert isinstance(afterwards, Generation)
    assert afterwards.reply == "the engine's" and len(brought.calls) == 2
    assert dict(engine.models) == {"strong": kept, "fast": fast}


async def test_an_engine_given_no_models_runs_on_the_runs(tmp_path: Path) -> None:
    run = await Engine(pair(tmp_path), models={}).run(
        "writer", "acme", "x", models={"fast": Scripted(text_reply("done"))}
    )
    assert isinstance(run, Generation) and run.reply == "done"


async def test_delegated_and_routed_runs_use_the_callers_models(tmp_path: Path) -> None:
    engine_strong, engine_judge, engine_fast = Scripted(), Answering(), Scripted()
    engine = Engine(
        crew(tmp_path),
        models={"strong": engine_strong, "judge": engine_judge, "fast": engine_fast},
    )
    lead = Scripted(
        delegating("writer"),
        delegating("triage", call_id="d2"),
        text_reply("done", model="lead-1"),
    )
    judge = NamedDecider("judge-1")
    fast = Scripted(text_reply("delegated", model="fast-1"), text_reply("routed", model="fast-2"))
    run = await engine.run(
        "lead", "acme", "x", models={"strong": lead, "judge": judge, "fast": fast}
    )
    assert isinstance(run, Generation)
    assert [step.is_error for step in run.steps] == [False, False]
    delegated, routing = run.children
    assert isinstance(delegated, Generation) and isinstance(routing, Decision)
    assert (run.model, delegated.model, routing.model) == ("lead-1", "fast-1", "judge-1")
    assert routing.routed_to is not None and routing.routed_to.model == "fast-2"
    assert len(lead.calls) == 3 and len(judge.calls) == 1 and len(fast.calls) == 2
    assert (engine_strong.calls, engine_judge.calls, engine_fast.calls) == ([], [], [])


async def test_concurrent_runs_keep_their_own_models(tmp_path: Path) -> None:
    step = asyncio.Barrier(2)

    def own(name: str) -> dict[str, Any]:
        return {
            "strong": Lockstep(f"lead-{name}", step, delegating("writer", name), text_reply(name)),
            "fast": Lockstep(f"writer-{name}", step, text_reply(f"drafted {name}")),
        }

    engine_strong, engine_fast = Scripted(), Scripted()
    engine = Engine(pair(tmp_path), models={"strong": engine_strong, "fast": engine_fast})
    one, other = own("one"), own("other")
    async with asyncio.timeout(5):
        runs = await asyncio.gather(
            engine.run("lead", "acme", "x", models=one),
            engine.run("lead", "acme", "x", models=other),
        )
    for run, name in zip(runs, ("one", "other"), strict=True):
        assert isinstance(run, Generation)
        assert (run.reply, run.model) == (name, f"lead-{name}")
        assert [m.model for m in run.conversation if m.role == "assistant"] == [f"lead-{name}"] * 2
        child = run.children[0]
        assert isinstance(child, Generation)
        assert (child.reply, child.model) == (f"drafted {name}", f"writer-{name}")
        assert not run.steps[0].is_error
    assert [adapter.calls for adapter in (*one.values(), *other.values())] == [2, 1, 2, 1]
    assert (engine_strong.calls, engine_fast.calls) == ([], [])


async def test_a_key_given_nowhere_fails_before_any_model_call(tmp_path: Path) -> None:
    given = Scripted()
    engine = Engine(crew(tmp_path), models={})
    with pytest.raises(
        ConfigurationError, match=f"^engine: agent 'lead' runs on model 'strong', {NOT_GIVEN}$"
    ):
        await engine.run("lead", "acme", "x", models={"fast": given})
    assert given.calls == []
    judge = Answering()
    with pytest.raises(
        ConfigurationError,
        match=f"^engine: agent 'triage' routes to agent 'writer', which runs on model 'fast', {NOT_GIVEN}$",
    ):
        await engine.run("triage", "acme", "x", models={"judge": judge})
    assert judge.calls == []


async def test_a_delegated_key_given_nowhere_reads_as_a_tool_error(tmp_path: Path) -> None:
    lead = Scripted(delegating("writer"), text_reply("noted"))
    run = await Engine(pair(tmp_path), models={}).run("lead", "acme", "x", models={"strong": lead})
    assert isinstance(run, Generation)
    assert run.steps[0].is_error
    assert f"agent 'writer' runs on model 'fast', {NOT_GIVEN}" in str(run.steps[0].result)


async def test_the_kind_is_checked_at_run_start(tmp_path: Path) -> None:
    engine = Engine(crew(tmp_path), models={})
    decider = Answering()
    with pytest.raises(
        ConfigurationError,
        match=r"^engine: agent 'lead' is a generation agent, and model 'strong' has no generate\(\)$",
    ):
        await engine.run("lead", "acme", "x", models={"strong": decider})
    assert decider.calls == []
    generator = Scripted()
    with pytest.raises(
        ConfigurationError,
        match=r"^engine: agent 'triage' is a decision agent, and model 'judge' has no decide\(\)$",
    ):
        await engine.run("triage", "acme", "x", models={"judge": generator})
    assert generator.calls == []


async def test_the_kind_is_checked_for_the_runs_agent_alone(tmp_path: Path) -> None:
    run = await Engine(crew(tmp_path), models={}).run(
        "lead",
        "acme",
        "x",
        models={"strong": Scripted(text_reply("ok")), "judge": Scripted(), "fast": Answering()},
    )
    assert isinstance(run, Generation) and run.reply == "ok"


async def test_a_delegated_or_routed_child_is_checked_at_its_own_start(tmp_path: Path) -> None:
    engine = Engine(crew(tmp_path), models={})
    lead = Scripted(delegating("triage"), text_reply("noted"))
    run = await engine.run("lead", "acme", "x", models={"strong": lead, "judge": Scripted()})
    assert isinstance(run, Generation)
    assert run.reply == "noted" and run.steps[0].is_error
    assert "agent 'triage' is a decision agent, and model 'judge' has no decide()" in str(
        run.steps[0].result
    )
    judge = Answering()
    with pytest.raises(
        ConfigurationError, match="agent 'writer' is a generation agent, and model 'fast' has no"
    ):
        await engine.run("triage", "acme", "x", models={"judge": judge, "fast": Answering()})
    assert len(judge.calls) == 1


async def test_run_model_records_what_the_reply_reports(tmp_path: Path) -> None:
    root = with_picture(tmp_path)
    engine = Engine(root, models={"fast": Named("engine-1"), "judge": NamedDecider("engine-2")})
    reported = await engine.run(
        "writer",
        "acme",
        "x",
        models={"fast": Named("declared-1", text_reply("a", model="reported-1"))},
    )
    assert reported.model == "reported-1"
    declared = await engine.run(
        "writer", "acme", "x", models={"fast": Named("declared-2", text_reply("b"))}
    )
    assert declared.model == "declared-2"
    decided = await engine.run("grader", "acme", "x", models={"judge": NamedDecider("declared-3")})
    assert decided.model == "declared-3"


async def test_media_types_follow_the_runs_model(tmp_path: Path) -> None:
    root = with_picture(tmp_path)
    taken = await Engine(root, models={"strong": Documents()}).run(
        "lead", "acme", "x", models={"strong": Pictures(LOAD, text_reply("seen"))}
    )
    assert isinstance(taken, Generation) and not taken.steps[0].is_error
    refused = await Engine(root, models={"strong": Pictures()}).run(
        "lead", "acme", "x", models={"strong": Documents(LOAD, text_reply("no"))}
    )
    assert isinstance(refused, Generation) and refused.steps[0].is_error
    assert "does not take image/png" in str(refused.steps[0].result)
    picture = [Image(media_type="image/png", data=PNG), Text(text="What is it?")]
    seen = await Engine(root, models={"judge": Answering()}).run(
        "grader", "acme", picture, models={"judge": Seeing()}
    )
    assert isinstance(seen, Decision) and seen.state == picture
    with pytest.raises(InputError, match="takes text only; got image/png"):
        await Engine(root, models={"judge": Seeing()}).run(
            "grader", "acme", picture, models={"judge": Answering()}
        )


async def test_replay_takes_models(tmp_path: Path, storage: Storage) -> None:
    root = with_picture(tmp_path)
    refinery = Refinery(storage)
    production = Engine(
        root,
        refinery=refinery,
        models={"fast": Scripted(text_reply("recorded")), "judge": Answering()},
    )
    await production.run("writer", "acme", "Draft it")
    await production.run("grader", "acme", "Is it urgent?")
    kept, kept_judge = Scripted(), Answering()
    candidate = Engine(root, models={"fast": kept, "judge": kept_judge})
    fast = Scripted(text_reply("replayed", model="run-1"))
    judge = NamedDecider("run-2")
    generated = [
        pair
        async for pair in refinery.replay("acme", candidate, agent="writer", models={"fast": fast})
    ]
    decided = [
        pair
        async for pair in refinery.replay(
            "acme", candidate, agent="grader", models={"judge": judge}
        )
    ]
    assert [pair.error for pair in (*generated, *decided)] == [None, None]
    assert generated[0].candidate is not None and generated[0].candidate.model == "run-1"
    assert decided[0].candidate is not None and decided[0].candidate.model == "run-2"
    assert kept.calls == [] and kept_judge.calls == []


async def test_a_model_only_a_mode_names_can_be_given_to_the_engine_or_the_run(
    tmp_path: Path,
) -> None:
    folder = with_a_mode(tmp_path)
    on_engine = await Engine(folder, models={"q": Scripted(text_reply("engine"))}).run(
        "a", "acme", "x", mode="quick"
    )
    assert isinstance(on_engine, Generation) and on_engine.reply == "engine"
    on_run = await Engine(folder, models={}).run(
        "a", "acme", "x", mode="quick", models={"q": Scripted(text_reply("run"))}
    )
    assert isinstance(on_run, Generation) and on_run.reply == "run"


def test_a_mode_on_a_model_of_the_other_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError, match=r"agent 'a' is a generation agent, and model 'q' has no generate"
    ):
        Engine(with_a_mode(tmp_path), models={"q": Answering()})


async def test_a_run_refuses_a_model_no_agent_runs_on(tmp_path: Path) -> None:
    llm = Scripted()
    engine = Engine(pair(tmp_path), models={"strong": llm})
    with pytest.raises(
        ConfigurationError,
        match="^engine: no agent runs on model 'fsat'; the tree's agents run on fast, strong$",
    ):
        await engine.run("lead", "acme", "x", models={"fsat": Scripted()})
    assert llm.calls == []


@pytest.mark.parametrize("models", UNFIT)
async def test_models_that_do_not_fit_are_refused_where_they_are_given(
    tmp_path: Path, storage: Storage, models: Any
) -> None:
    engine = Engine(pair(tmp_path), models={})
    with pytest.raises(ConfigurationError, match="^engine: "):
        await engine.run("lead", "acme", "x", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        engine.tool("lead", scope="acme", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        EngineTools(engine, agent="lead", scope="acme", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        Refinery(storage).replay("acme", engine, agent="lead", models=models)


async def test_engine_tool_models_reach_its_run_and_what_it_delegates(tmp_path: Path) -> None:
    engine = Engine(pair(tmp_path), models={"strong": Scripted(), "fast": Scripted()})
    lead = Scripted(delegating("writer"), text_reply("done"))
    fast = Scripted(text_reply("drafted"))
    tools = engine.tool("lead", scope="acme", models={"strong": lead, "fast": fast})
    assert await tools.dispatch("lead", {"input": "x"}) == "done"
    assert len(lead.calls) == 2 and len(fast.calls) == 1


async def test_a_run_lends_no_models_to_an_engine_tool_it_calls(tmp_path: Path) -> None:
    engine = Engine(pair(tmp_path), models={})
    consult = engine.tool("writer", scope="acme", name="consult")
    lead = Scripted(
        Reply(
            tool_calls=[ToolCall(id="c1", name="consult", args={"input": "x"})],
            stop_reason="tool_use",
        ),
        text_reply("noted"),
    )
    fast = Scripted()
    run = await engine.run(
        "lead", "acme", "x", tools=consult, models={"strong": lead, "fast": fast}
    )
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert f"agent 'writer' runs on model 'fast', {NOT_GIVEN}" in str(run.steps[0].result)
    assert fast.calls == []


async def test_engine_tools_follow_the_models_given(tmp_path: Path) -> None:
    engine = Engine(with_picture(tmp_path), models={"strong": Documents()})
    loaded = await EngineTools(
        engine, agent="lead", scope="acme", models={"strong": Pictures()}
    ).dispatch("Load", {"path": "diagram.png"})
    assert loaded
    with pytest.raises(InputError, match="does not take image/png"):
        await EngineTools(engine, agent="lead", scope="acme").dispatch(
            "Load", {"path": "diagram.png"}
        )
    fast = Scripted(text_reply("drafted"))
    delegated = await EngineTools(
        engine, agent="lead", scope="acme", models={"fast": fast}
    ).dispatch("Delegate", {"agent": "writer", "input": "x"})
    assert delegated == "drafted" and len(fast.calls) == 1
