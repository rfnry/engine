from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Message
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool

from .fakes import Answering, Scripted, delegating, text_reply

PNG = b"\x89PNG\r\n\x1a\ndiagram"


class Pictures(Scripted):
    media_types = {"image/*"}


class Documents(Scripted):
    media_types = {"application/pdf"}


class Named(Scripted):
    def __init__(self, name: str, *replies: Any) -> None:
        super().__init__(*replies)
        self.model = name


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def pair(root: Path) -> Path:
    write(root / "lead/AGENT.md", agent_md("lead", extra="agents: [writer]\n"))
    write(root / "writer/AGENT.md", agent_md("writer", model="fast"))
    return root


async def noop(args: dict[str, Any]) -> str:
    return "ok"


async def test_each_agent_runs_on_the_model_its_agent_md_names(tmp_path: Path) -> None:
    strong = Scripted(delegating("writer"), text_reply("done", model="strong-1"))
    fast = Scripted(text_reply("drafted", model="fast-1"))
    run = await Engine(pair(tmp_path), models={"strong": strong, "fast": fast}).run(
        "lead", "acme", "x"
    )
    assert isinstance(run, Generation)
    assert (run.reply, run.model, run.children[0].model) == ("done", "strong-1", "fast-1")
    child = run.children[0]
    assert isinstance(child, Generation)
    assert [m.model for m in child.conversation if m.role == "assistant"] == ["fast-1"]
    assert len(strong.calls) == 2 and len(fast.calls) == 1


def test_a_model_no_agent_runs_on_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="no agent runs on model 'fsat'; the tree's agents run on fast, strong",
    ):
        Engine(
            pair(tmp_path), models={"strong": Scripted(), "fast": Scripted(), "fsat": Scripted()}
        )


def test_an_agent_on_a_model_of_the_other_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="agent 'writer' is a generation agent, and model 'fast' has no generate",
    ):
        Engine(pair(tmp_path), models={"strong": Scripted(), "fast": Answering()})


def test_a_decision_agent_on_a_generation_model_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    with pytest.raises(
        ConfigurationError,
        match="agent 'triage' is a decision agent, and model 'judge' has no decide",
    ):
        Engine(tmp_path, models={"judge": Scripted()})


@pytest.mark.parametrize(
    "models",
    [
        [("strong", 1)],
        {"": Scripted()},
        {" strong": Scripted()},
        {1: Scripted()},
        {"strong": object()},
    ],
)
def test_a_models_mapping_that_does_not_fit_is_refused(tmp_path: Path, models: Any) -> None:
    with pytest.raises(ConfigurationError, match="^engine: "):
        Engine(pair(tmp_path), models=models)


async def test_an_agent_whose_model_was_not_given_is_refused_when_it_runs_while_the_others_run(
    tmp_path: Path,
) -> None:
    engine = Engine(pair(tmp_path), models={"strong": Scripted(text_reply("alone"))})
    with pytest.raises(
        ConfigurationError,
        match="^engine: agent 'writer' runs on model 'fast', which this engine was not given$",
    ):
        await engine.run("writer", "acme", "x")
    alone = await engine.run("lead", "acme", "x")
    assert isinstance(alone, Generation) and alone.reply == "alone"


async def test_a_delegate_to_an_agent_whose_model_was_not_given_reads_as_a_tool_error(
    tmp_path: Path,
) -> None:
    engine = Engine(
        pair(tmp_path), models={"strong": Scripted(delegating("writer"), text_reply("ok"))}
    )
    run = await engine.run("lead", "acme", "x")
    assert isinstance(run, Generation)
    step = run.steps[0]
    assert step.is_error and "runs on model 'fast', which this engine was not given" in str(
        step.result
    )


async def test_a_router_is_refused_before_its_questions_when_a_target_s_model_was_not_given(
    tmp_path: Path,
) -> None:
    write(
        tmp_path / "triage/AGENT.md",
        agent_md(
            "triage",
            kind="decision",
            model="judge",
            extra="route:\n  by: team\n  to: {writing: writer}\n",
        ),
    )
    write(
        tmp_path / "triage/questions/team.md",
        "---\ntype: choice\noptions: {writing: Writing, other: Other}\n---\nWhich?\n",
    )
    write(tmp_path / "writer/AGENT.md", agent_md("writer", model="fast"))
    judge = Answering()
    engine = Engine(tmp_path, models={"judge": judge})
    with pytest.raises(
        ConfigurationError,
        match="routes to agent 'writer', which runs on model 'fast', which this engine was not given",
    ):
        await engine.run("triage", "acme", "x")
    assert judge.calls == []


def with_picture(root: Path) -> Path:
    write(root / "lead/AGENT.md", agent_md("lead", extra="preload: true\n"))
    (root / "lead/diagram.png").write_bytes(PNG)
    write(root / "writer/AGENT.md", agent_md("writer", model="fast"))
    return root


def test_an_agent_s_preload_follows_its_own_model_s_media_types(tmp_path: Path) -> None:
    root = with_picture(tmp_path)
    engine = Engine(root, models={"strong": Pictures(), "fast": Documents()})
    assert engine.agents["lead"].model == "strong"
    write(root / "writer/AGENT.md", agent_md("writer", model="fast", extra="preload: true\n"))
    (root / "writer/diagram.png").write_bytes(PNG)
    with pytest.raises(ConfigurationError, match="agent 'writer' cannot preload"):
        Engine(root, models={"strong": Pictures(), "fast": Documents()})


async def test_load_follows_the_media_types_of_the_agent_s_model(tmp_path: Path) -> None:
    root = with_picture(tmp_path)
    (root / "writer/diagram.png").write_bytes(PNG)
    engine = Engine(root, models={"strong": Pictures(), "fast": Documents()})
    assert await EngineTools(engine, agent="lead", scope="acme").dispatch(
        "Load", {"path": "diagram.png"}
    )
    with pytest.raises(InputError, match="does not take image/png"):
        await EngineTools(engine, agent="writer", scope="acme").dispatch(
            "Load", {"path": "diagram.png"}
        )


def declaring(root: Path) -> Path:
    pair(root)
    write(root / "writer/AGENT.md", agent_md("writer", model="fast", extra="tools: [fetch]\n"))
    return root


def test_declared_tools_need_bodies_only_for_agents_whose_model_was_given(tmp_path: Path) -> None:
    root = declaring(tmp_path)
    Engine(root, models={"strong": Scripted()})
    with pytest.raises(ConfigurationError, match="fetch"):
        Engine(root, models={"strong": Scripted(), "fast": Scripted()})
    Engine(
        root,
        models={"strong": Scripted(), "fast": Scripted()},
        tools=[Tool("fetch", "Fetch a page", body=noop)],
    )


def test_engine_models_is_the_mapping_given_and_there_is_no_generation(tmp_path: Path) -> None:
    strong, fast = Scripted(), Scripted()
    engine = Engine(pair(tmp_path), models={"strong": strong, "fast": fast})
    assert engine.models == {"strong": strong, "fast": fast}
    assert not hasattr(engine, "generation") and not hasattr(engine, "decision")


async def test_the_fingerprint_is_the_same_whichever_adapter_a_name_maps_to(
    tmp_path: Path,
) -> None:
    root = pair(tmp_path)
    prints = []
    for strong in (Scripted(text_reply("a")), Named("other", text_reply("b"))):
        engine = Engine(root, models={"strong": strong, "fast": Pictures()})
        run = await engine.run("lead", "acme", "x")
        assert isinstance(run, Generation)
        prints.append(run.fingerprint.agent)
    assert prints[0] == prints[1]


async def test_reasoning_does_not_cross_to_an_agent_on_another_model(tmp_path: Path) -> None:
    thought = {"thought": "plan"}
    strong = Named("model-a", text_reply("planned", reasoning=thought))
    fast = Named("model-b", text_reply("written"))
    root = pair(tmp_path)
    first = await Engine(root, models={"strong": strong, "fast": Named("model-b")}).run(
        "lead", "acme", "x"
    )
    assert isinstance(first, Generation)
    history: list[Message] = first.conversation
    assert any(m.reasoning == thought for m in history)
    await Engine(root, models={"strong": Named("model-a"), "fast": fast}).run(
        "writer", "acme", "y", history=history
    )
    sent = [m for m in fast.calls[0][1] if m.role == "assistant"]
    assert sent and all(m.reasoning is None for m in sent)
