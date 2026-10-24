import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import DelegationLoopError, InputError, NotFoundError
from rfnry_engine.models import (
    Answer,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    QuestionLike,
    Reply,
    ScoreAnswer,
    State,
    ToolCall,
)
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.delegate import QUESTION
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import (
    PICK,
    Answering,
    Desk,
    Fixed,
    Scripted,
    delegate_triage,
    delegating,
    lead,
    text_reply,
)

TRIAGE: dict[str, Answer] = {
    "urgent": CheckAnswer(probability=0.9),
    "severity": ScoreAnswer(score=3, probabilities=[0, 0, 0, 1]),
}


def _delegate(agent: str, text: str, call_id: str = "d1") -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=call_id, name="Delegate", args={"agent": agent, "input": text})],
        stop_reason="tool_use",
    )


async def test_delegate_tool_exists_only_when_agents_are_named(fixtures: Path) -> None:
    llm = Scripted(text_reply("a"), text_reply("b"))
    engine = Engine(fixtures, models={"strong": llm})
    await engine.run("support", "acme", "x")
    await engine.run("billing", "acme", "x")
    assert "Delegate" in [t.name for t in llm.calls[0][2]]
    assert "Delegate" not in [t.name for t in llm.calls[1][2]]
    spec = next(t for t in llm.calls[0][2] if t.name == "Delegate")
    assert spec.input_schema["properties"]["agent"]["enum"] == ["triage"]
    assert list(spec.input_schema["properties"]) == ["agent", "input", "questions"]
    assert "- triage (decision agent): Judges urgency" in spec.description
    assert "Delegate" in EngineTools(engine, agent="support", scope="acme")
    assert "Delegate" not in EngineTools(engine, agent="billing", scope="acme")


async def test_generation_agent_consults_a_decision_agent(fixtures: Path) -> None:
    llm = Scripted(_delegate("triage", "It is down"), text_reply("Escalating."))
    engine = Engine(fixtures, models={"strong": llm, "judge": Fixed(TRIAGE)})
    run = await engine.run("support", "acme", "It is down")
    assert isinstance(run, Generation) and len(run.children) == 1
    child = run.children[0]
    assert (
        isinstance(child, Decision)
        and child.agent == "triage"
        and child.parent_run_id == run.run_id
    )
    assert run.steps[0].child_run_id == child.run_id
    assert json.loads(str(run.steps[0].result))["answers"]["urgent"]["probability"] == 0.9


async def test_the_whole_chain_frontdesk_support_triage(fixtures: Path) -> None:
    intent = {
        "intent": ChoiceAnswer(
            choice="technical", probabilities={"technical": 0.9, "billing": 0.05, "other": 0.05}
        )
    }
    answers: list[Any] = [intent, TRIAGE]
    dm = Fixed(lambda state: answers.pop(0))
    llm = Scripted(_delegate("triage", "It is down"), text_reply("Handled."))
    engine = Engine(fixtures, models={"strong": llm, "judge": dm})
    run = await engine.run("frontdesk", "acme", "It is down")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.agent == "support" and run.routed_to.children[0].agent == "triage"


async def test_an_agent_delegates_to_a_fresh_run_of_itself(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\nagents: [a]\n---\nhi\n"
    )
    llm = Scripted(_delegate("a", "the sub-problem"), text_reply("sub done"), text_reply("done"))
    run = await Engine(tmp_path, models={"strong": llm}).run("a", "acme", "start")
    assert isinstance(run, Generation) and run.reply == "done"
    child = run.children[0]
    assert isinstance(child, Generation) and child.agent == "a" and child.reply == "sub done"
    [delegate] = [t for t in llm.calls[0][2] if t.name == "Delegate"]
    assert delegate.description is not None and "\n- a: " in delegate.description


async def test_a_delegation_cycle_runs_until_max_depth(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        other = "b" if name == "a" else "a"
        (tmp_path / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: generation\nmodel: strong\nagents: [{other}]\n---\nhi\n"
        )
    llm = Scripted(
        _delegate("b", "go"), _delegate("a", "back"), text_reply("b done"), text_reply("a done")
    )
    run = await Engine(tmp_path, max_depth=2, models={"strong": llm}).run("a", "acme", "start")
    assert isinstance(run, Generation)
    child = run.children[0]
    assert isinstance(child, Generation) and child.steps[0].is_error
    assert (
        child.steps[0].result == "DelegationDepthError: delegate: depth 2 exceeded at a -> b -> a"
    )


async def test_a_route_back_into_the_chain_raises(tmp_path: Path) -> None:
    for name, other in (("r1", "r2"), ("r2", "r1")):
        (tmp_path / name / "questions").mkdir(parents=True)
        (tmp_path / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: decision\nmodel: judge\nroute: {{by: go, to: {{next: {other}}}}}\n---\n"
        )
        (tmp_path / name / "questions" / "go.md").write_text(
            "---\ntype: choice\noptions: {stay: Keep it, next: Pass it on}\n---\nWhere?\n"
        )
    engine = Engine(tmp_path, models={"judge": Answering()})
    with pytest.raises(DelegationLoopError, match=r"^delegate: loop at r1 -> r2 -> r1$"):
        await engine.run("r1", "acme", "x")


def _router(tmp_path: Path, name: str, target: str) -> None:
    (tmp_path / name / "questions").mkdir(parents=True)
    (tmp_path / name / "AGENT.md").write_text(
        f"---\nname: {name}\nkind: decision\nmodel: judge\nroute: {{by: go, to: {{next: {target}}}}}\n---\n"
    )
    (tmp_path / name / "questions" / "go.md").write_text(
        "---\ntype: choice\noptions: {stay: Keep it, next: Pass it on}\n---\nWhere?\n"
    )


async def test_a_router_reached_through_delegate_may_route_back_to_the_agent_that_delegated(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\nagents: [r]\n---\nhi\n"
    )
    _router(tmp_path, "r", "a")
    llm = Scripted(_delegate("r", "the sub-problem"), text_reply("sub done"), text_reply("done"))
    run = await Engine(tmp_path, models={"strong": llm, "judge": Answering()}).run(
        "a", "acme", "start"
    )
    assert isinstance(run, Generation) and run.reply == "done"
    assert not any(step.is_error for step in run.steps)
    [router] = run.children
    assert isinstance(router, Decision) and router.agent == "r"
    routed = router.routed_to
    assert isinstance(routed, Generation) and routed.agent == "a" and routed.reply == "sub done"


async def test_a_route_loop_below_a_delegate_is_still_refused(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\nagents: [r1]\n---\nhi\n"
    )
    _router(tmp_path, "r1", "r2")
    _router(tmp_path, "r2", "r1")
    llm = Scripted(_delegate("r1", "go"), text_reply("a done"))
    run = await Engine(tmp_path, models={"strong": llm, "judge": Answering()}).run(
        "a", "acme", "start"
    )
    assert isinstance(run, Generation) and run.reply == "a done"
    assert run.steps[0].is_error
    assert run.steps[0].result == "DelegationLoopError: delegate: loop at a -> r1 -> r2 -> r1"
    told = llm.calls[1][1][-1]
    assert told.role == "tool" and told.is_error and told.content == run.steps[0].result


async def test_depth_is_capped(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\nagents: [b]\n---\nhi\n"
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "AGENT.md").write_text(
        "---\nname: b\nkind: generation\nmodel: strong\n---\nhi\n"
    )
    llm = Scripted(_delegate("b", "go"), text_reply("a done"))
    run = await Engine(tmp_path, max_depth=1, models={"strong": llm}).run("a", "acme", "start")
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert "DelegationDepthError" in run.steps[0].result and run.children == []


async def test_the_child_serves_the_same_tenant_with_its_context(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\nagents: [b]\n---\nhi\n"
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "AGENT.md").write_text(
        "---\nname: b\nkind: generation\nmodel: strong\n---\nhi\n"
    )
    llm = Scripted(_delegate("b", "go"), text_reply("b done"), text_reply("a done"))
    run = await Engine(tmp_path, models={"strong": llm}).run(
        "a", "acme", "start", context="Acme is enterprise."
    )
    assert isinstance(run, Generation)
    child = run.children[0]
    assert isinstance(child, Generation) and child.scope == "acme"
    assert "Acme is enterprise." in child.system
    assert run.steps[0].result == "b done"


async def test_dispatch_delegate_outside_a_run(fixtures: Path) -> None:
    engine = Engine(fixtures, models={"strong": Scripted(), "judge": Fixed(TRIAGE)})
    tools = EngineTools(engine, agent="support", scope="acme")
    result = await tools.dispatch("Delegate", {"agent": "triage", "input": "It is down"})
    assert isinstance(result, str) and json.loads(result)["answers"]["urgent"]["probability"] == 0.9


async def test_dispatch_delegate_asks_a_decision_agent_the_questions_it_is_given(
    tmp_path: Path,
) -> None:
    class Recorded:
        def __init__(self) -> None:
            self.runs: list[Any] = []

        async def record(self, run: Any) -> None:
            self.runs.append(run)

    recorder = Recorded()
    dm = Answering()
    engine = Engine(lead(tmp_path), refinery=recorder, models={"strong": Scripted(), "judge": dm})
    tools = EngineTools(engine, agent="lead", scope="acme")
    result = await tools.dispatch(
        "Delegate", {"agent": "blank", "input": "x", "questions": {"area": PICK}}
    )
    assert json.loads(str(result))["answers"]["area"]["choice"] == "orders"
    assert dm.calls == [("x", {"area": Choice.model_validate(PICK)})]
    [run] = recorder.runs
    assert isinstance(run, Decision) and run.agent == "blank" and run.runtime == ["area"]


async def test_dispatch_delegate_gives_the_child_the_tenant_tools(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\nagents: [b]\n---\nhi\n"
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "AGENT.md").write_text(
        "---\nname: b\nkind: generation\nmodel: strong\n---\nhi\n"
    )

    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    pinned = Tools([Tool("ping", "Ping", body=ping)])
    llm = Scripted(text_reply("b done"))
    tools = EngineTools(
        Engine(tmp_path, models={"strong": llm}), agent="a", scope="acme", tools=pinned
    )
    assert await tools.dispatch("Delegate", {"agent": "b", "input": "go"}) == "b done"
    assert "ping" in [t.name for t in llm.calls[0][2]]


async def test_delegate_is_not_bound_by_the_tool_deadline(fixtures: Path) -> None:
    class Slow(Fixed):
        async def decide(self, state: Any, questions: Any) -> Any:
            await asyncio.sleep(0.2)
            return TRIAGE

    llm = Scripted(_delegate("triage", "It is down"), text_reply("Escalating."))
    engine = Engine(fixtures, tool_timeout=0.05, models={"strong": llm, "judge": Slow(TRIAGE)})
    run = await engine.run("support", "acme", "It is down")
    assert isinstance(run, Generation) and not run.steps[0].is_error
    assert run.children[0].agent == "triage"


BAD_DELEGATIONS: list[tuple[dict[str, Any], type[Exception], str]] = [
    (
        {"agent": "writer", "input": "x", "task": "ghost"},
        NotFoundError,
        "engine: agent 'writer' has no task 'ghost'",
    ),
    (
        {"agent": "grader", "input": "x", "task": "apology"},
        NotFoundError,
        "engine: agent 'grader' has no task 'apology'",
    ),
    (
        {"agent": "writer", "input": "x", "questions": {"area": PICK}},
        InputError,
        "engine: generation agent 'writer' takes no questions",
    ),
    (
        {"agent": "blank", "input": "x", "questions": {"area": {**PICK, "options": {"a": None}}}},
        InputError,
        "engine: questions do not fit the contract: ",
    ),
    (
        {"agent": "blank", "input": "x", "questions": {"area": {"type": "check"}}},
        InputError,
        "engine: questions do not fit the contract: ",
    ),
    (
        {"agent": "grader", "input": "x", "task": "tone", "questions": {"polite": PICK}},
        InputError,
        "engine: decision agent 'grader' already asks 'polite' from its folder",
    ),
    (
        {"agent": "blank", "input": "x"},
        InputError,
        "engine: decision agent 'blank' has no questions to ask; give it questions/ or tasks/<name>/, or ask it through Delegate",
    ),
    (
        {"agent": "blank", "input": "x", "questions": "Which area?"},
        InputError,
        "delegate: invalid arguments: ",
    ),
    (
        {"agent": "support", "input": "x"},
        InputError,
        "delegate: agent 'lead' may not delegate to 'support'",
    ),
    ({"agent": "blank"}, InputError, "delegate: invalid arguments: "),
]


@pytest.mark.parametrize(("args", "error", "message"), BAD_DELEGATIONS)
async def test_a_refused_delegation_is_a_tool_error_the_same_in_a_run_and_from_code(
    tmp_path: Path, args: dict[str, Any], error: type[Exception], message: str
) -> None:
    root = lead(tmp_path)
    llm = Scripted(delegating(**args) if "input" in args else _bare(args), text_reply("done"))
    dm = Answering()
    engine = Engine(root, models={"strong": llm, "judge": dm})
    run = await engine.run("lead", "acme", "x")
    assert isinstance(run, Generation) and run.reply == "done" and run.children == []
    assert run.steps[0].is_error and run.steps[0].child_run_id is None
    assert str(run.steps[0].result).startswith(f"{error.__name__}: {message}")
    with pytest.raises(error) as raised:
        await EngineTools(engine, agent="lead", scope="acme").dispatch("Delegate", args)
    assert run.steps[0].result == f"{error.__name__}: {raised.value}"
    assert dm.calls == [] and len(llm.calls) == 2


def _bare(args: dict[str, Any]) -> Reply:
    return Reply(tool_calls=[ToolCall(id="d", name="Delegate", args=args)], stop_reason="tool_use")


async def test_delegate_offers_a_task_and_questions_only_when_a_target_takes_them(
    tmp_path: Path,
) -> None:
    root = lead(tmp_path)
    (root / "plain/AGENT.md").parent.mkdir()
    (root / "plain/AGENT.md").write_text(
        "---\nname: plain\nkind: generation\nmodel: strong\ndescription: Plain\nagents: [writer]\n---\nHi\n"
    )
    engine = Engine(root, models={"strong": Scripted(), "judge": Answering()})
    [offered] = EngineTools(engine, agent="lead", scope="acme").only("Delegate").definitions()
    schema = offered["input_schema"]
    assert schema["required"] == ["agent", "input"]
    assert schema["properties"]["task"] == {
        "type": "string",
        "enum": ["tone", "apology"],
        "description": "A task of the agent, to run it with",
    }
    assert schema["properties"]["questions"]["additionalProperties"] == QUESTION
    assert offered["description"] == (
        "Hand an input to another agent and get its reply.\n"
        "- grader (decision agent; tasks: tone): Grades replies\n"
        "- writer (tasks: apology): Writes replies\n"
        "- blank (decision agent): Answers what it is asked"
    )
    [plain] = EngineTools(engine, agent="plain", scope="acme").only("Delegate").definitions()
    assert list(plain["input_schema"]["properties"]) == ["agent", "input", "task"]


async def test_an_adapter_that_edits_the_offered_schema_changes_no_later_run(
    tmp_path: Path,
) -> None:
    engine = Engine(lead(tmp_path), models={"strong": Scripted(), "judge": Answering()})
    [first] = EngineTools(engine, agent="lead", scope="acme").only("Delegate").definitions()
    asked = first["input_schema"]["properties"]["questions"]["additionalProperties"]
    asked["additionalProperties"] = False
    asked["properties"]["levels"]["items"]["type"].append("number")
    [second] = EngineTools(engine, agent="lead", scope="acme").only("Delegate").definitions()
    assert second["input_schema"]["properties"]["questions"]["additionalProperties"] == QUESTION
    assert "additionalProperties" not in QUESTION
    assert QUESTION["properties"]["levels"]["items"]["type"] == ["string", "object", "array"]


async def test_a_generation_agent_asks_a_decision_agent_a_question_it_composed(
    tmp_path: Path,
) -> None:
    llm = Scripted(delegating("blank", "Charged twice", questions={"area": PICK}), text_reply("ok"))
    dm = Answering()
    run = await Engine(lead(tmp_path), models={"strong": llm, "judge": dm}).run("lead", "acme", "x")
    assert isinstance(run, Generation) and not run.steps[0].is_error
    [child] = run.children
    assert isinstance(child, Decision) and child.agent == "blank" and child.runtime == ["area"]
    assert child.questions == {"area": Choice.model_validate(PICK)} and child.task is None
    assert json.loads(str(run.steps[0].result))["answers"]["area"]["choice"] == "orders"
    assert run.steps[0].args["questions"] == {"area": PICK}
    assert dm.calls == [("Charged twice", {"area": Choice.model_validate(PICK)})]


async def test_delegate_names_a_task_for_either_kind_and_adds_questions_to_it(
    tmp_path: Path,
) -> None:
    angry = {"type": "check", "instructions": "Is `message` angry?", "yes": ["Insults", "Caps"]}
    llm = Scripted(
        Reply(
            tool_calls=[
                *delegating(
                    "grader", "Sorry!", "g", task="tone", questions={"angry": angry}
                ).tool_calls,
                *delegating("writer", "Late parcel", "w", task="apology").tool_calls,
            ],
            stop_reason="tool_use",
        ),
        text_reply("Apologised."),
        text_reply("done"),
    )
    run = await Engine(lead(tmp_path), models={"strong": llm, "judge": Answering()}).run(
        "lead", "acme", "x"
    )
    assert isinstance(run, Generation) and [s.is_error for s in run.steps] == [False, False]
    graded, written = run.children
    assert isinstance(graded, Decision) and graded.task == "tone" and graded.runtime == ["angry"]
    assert list(graded.questions) == ["urgent", "polite", "angry"]
    assert graded.questions["angry"] == Check.model_validate(angry)
    assert isinstance(written, Generation) and written.task == "apology"
    assert "<<<TASK>>>\nApologise first." in written.system and written.reply == "Apologised."


async def test_children_follow_their_steps_whatever_order_they_finish_in(fixtures: Path) -> None:
    finished: list[State] = []

    class SlowFirst(Desk):
        async def decide(
            self, state: State, questions: Mapping[str, QuestionLike]
        ) -> dict[str, Answer]:
            if state == "slow":
                await asyncio.sleep(0.02)
            finished.append(state)
            return await super().decide(state, questions)

    both = delegate_triage("d1", "slow").tool_calls + delegate_triage("d2", "fast").tool_calls
    llm = Scripted(Reply(tool_calls=both, stop_reason="tool_use"), text_reply("Both triaged."))
    engine = Engine(fixtures, models={"strong": llm, "judge": SlowFirst()})
    run = await engine.run("support", "acme", "Two tickets")
    assert isinstance(run, Generation) and finished == ["fast", "slow"]
    assert [child.run_id for child in run.children] == [step.child_run_id for step in run.steps]
    assert [child.state for child in run.children if isinstance(child, Decision)] == [
        "slow",
        "fast",
    ]
