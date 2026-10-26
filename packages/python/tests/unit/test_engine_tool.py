import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import (
    Answer,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Reply,
    ScoreAnswer,
    ToolCall,
)
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.tools.delegate import QUESTION, child_result
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import NO_QUESTIONS, PICK, Answering, Fixed, Scripted, lead, text_reply

TRIAGE: dict[str, Answer] = {
    "urgent": CheckAnswer(probability=0.9),
    "severity": ScoreAnswer(score=3, probabilities=[0, 0, 0, 1]),
}


class Recorded:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


def calling(name: str, text: str) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id="c1", name=name, args={"input": text})], stop_reason="tool_use"
    )


async def test_a_generation_agent_as_a_tool_replies_and_records(fixtures: Path) -> None:
    llm = Scripted(text_reply("Refunded."))
    recorder = Recorded()
    engine = Engine(fixtures, refinery=recorder, models={"strong": llm})
    billing = engine.tool("billing", scope="acme", context="Acme is on the enterprise plan.")
    [definition] = billing.definitions()
    assert (
        definition["name"] == "billing" and definition["description"] == "Answers billing questions"
    )
    assert definition["input_schema"]["required"] == ["input"]
    assert await billing.dispatch("billing", {"input": "Charged twice"}) == "Refunded."
    [run] = recorder.runs
    assert isinstance(run, Generation) and run.agent == "billing" and run.scope == "acme"
    assert run.parent_run_id is None and run.input == "Charged twice"
    assert "Acme is on the enterprise plan." in llm.calls[0][0]


async def test_a_decision_agent_as_a_tool_returns_its_answers(fixtures: Path) -> None:
    engine = Engine(fixtures, models={"judge": Fixed(TRIAGE)})
    judge = engine.tool("triage", scope="acme", name="judge")
    assert judge.names == ["judge"]
    result = await judge.dispatch("judge", {"input": "It is down"})
    assert isinstance(result, dict) and list(result) == ["answers"]
    answers = result["answers"]
    assert set(answers) == {"urgent", "severity"}
    assert answers["urgent"]["probability"] == 0.9 and answers["severity"]["score"] == 3


async def test_the_tool_composes_and_refuses_bad_calls(fixtures: Path) -> None:
    engine = Engine(fixtures, models={"strong": Scripted()})

    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    surface = engine.tool("billing", scope="acme", name="consult") + Tools(
        [Tool("ping", "Ping", body=ping)]
    )
    assert surface.names == ["consult", "ping"]
    with pytest.raises(InputError, match="^tools: invalid arguments for consult"):
        await surface.dispatch("consult", {})
    with pytest.raises(ConfigurationError, match="^tools: duplicate tool name 'consult'"):
        surface + engine.tool("support", scope="acme", name="consult")
    with pytest.raises(InputError, match="^engine: scope is required"):
        engine.tool("billing", scope="")
    with pytest.raises(NotFoundError, match="^engine: agent 'ghost' was not found"):
        engine.tool("ghost", scope="acme")


async def test_inside_a_run_it_is_its_own_top_level_run(fixtures: Path) -> None:
    llm = Scripted(
        calling("consult", "Charged twice"), text_reply("Refunded."), text_reply("Done.")
    )
    recorder = Recorded()
    engine = Engine(fixtures, refinery=recorder, models={"strong": llm})
    run = await engine.run(
        "support", "acme", "Help", tools=engine.tool("billing", scope="acme", name="consult")
    )
    assert isinstance(run, Generation) and run.reply == "Done." and run.children == []
    assert run.steps[0].result == "Refunded." and run.steps[0].child_run_id is None
    assert [r.agent for r in recorder.runs] == ["billing", "support"]
    assert recorder.runs[0].parent_run_id is None


async def test_the_depth_guard_holds_across_the_tool(fixtures: Path) -> None:
    llm = Scripted(calling("billing", "again"), text_reply("done"))
    engine = Engine(fixtures, max_depth=1, models={"strong": llm})
    run = await engine.run("billing", "acme", "hi", tools=engine.tool("billing", scope="acme"))
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert run.steps[0].result == (
        "DelegationDepthError: tools/delegate: depth 1 exceeded at billing -> billing"
    )


async def test_a_call_through_the_tool_resets_the_route_chain(tmp_path: Path) -> None:
    (tmp_path / "r" / "questions").mkdir(parents=True)
    (tmp_path / "r" / "AGENT.md").write_text(
        "---\nname: r\nkind: decision\nmodel: judge\ndescription: Passes it on\nroute: {by: go, to: {next: g}}\n---\n"
    )
    (tmp_path / "r" / "questions" / "go.json").write_text(
        '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep it", "next": "Pass it on"}}'
    )
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "AGENT.md").write_text(
        "---\nname: g\nkind: generation\nmodel: strong\n---\nhi\n"
    )
    llm = Scripted(calling("r", "again"), text_reply("inner done"), text_reply("outer done"))
    engine = Engine(tmp_path, models={"strong": llm, "judge": Answering()})
    run = await engine.run("r", "acme", "start", tools=engine.tool("r", scope="acme"))
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    [step] = run.routed_to.steps
    assert not step.is_error and run.routed_to.reply == "outer done"
    assert isinstance(step.result, str)
    assert json.loads(step.result)["routed_to"] == {"agent": "g", "reply": "inner done"}


async def test_a_decision_agent_as_a_tool_takes_questions_of_the_caller(
    tmp_path: Path,
) -> None:
    recorder = Recorded()
    engine = Engine(lead(tmp_path), refinery=recorder, models={"judge": Answering()})
    grader = engine.tool("grader", scope="acme")
    [definition] = grader.definitions()
    schema = definition["input_schema"]
    assert list(schema["properties"]) == ["input", "questions"]
    assert schema["required"] == ["input"]
    assert schema["properties"]["questions"]["additionalProperties"] == QUESTION
    result = await grader.dispatch("grader", {"input": "Thanks!", "questions": {"area": PICK}})
    assert isinstance(result, dict)
    answers = result["answers"]
    assert list(answers) == ["urgent", "area"]
    assert answers["area"]["choice"] == "orders"
    [run] = recorder.runs
    assert isinstance(run, Decision) and run.runtime == ["area"]
    assert run.questions["area"] == Choice.model_validate(PICK)
    with pytest.raises(InputError, match="^engine: decision agent 'grader' takes no task; "):
        await grader.dispatch("grader", {"input": "Thanks!", "task": "tone"})
    blank = engine.tool("blank", scope="acme")
    assert list(blank.definitions()[0]["input_schema"]["properties"]) == ["input", "questions"]
    asked = await blank.dispatch("blank", {"input": "x", "questions": {"area": PICK}})
    assert isinstance(asked, dict) and set(asked["answers"]) == {"area"}
    with pytest.raises(InputError, match="^engine: questions do not fit the contract: "):
        await blank.dispatch("blank", {"input": "x", "questions": {"area": {"type": "score"}}})
    with pytest.raises(
        InputError,
        match=f"^{NO_QUESTIONS}$",
    ):
        await blank.dispatch("blank", {"input": "x"})


async def test_a_generation_agent_as_a_tool_takes_a_task_it_has(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Sorry, it is late."))
    recorder = Recorded()
    engine = Engine(lead(tmp_path), refinery=recorder, models={"strong": llm, "judge": Answering()})
    writer = engine.tool("writer", scope="acme")
    properties = writer.definitions()[0]["input_schema"]["properties"]
    assert list(properties) == ["input", "task"] and properties["task"]["enum"] == ["apology"]
    await writer.dispatch("writer", {"input": "Late parcel", "task": "apology"})
    [run] = recorder.runs
    assert isinstance(run, Generation) and run.task == "apology"
    assert "<<<TASK>>>\nApologise first." in llm.calls[0][0]
    with pytest.raises(NotFoundError, match="^engine: agent 'writer' has no task 'tone'; "):
        await writer.dispatch("writer", {"input": "x", "task": "tone"})
    with pytest.raises(InputError, match="^engine: generation agent 'writer' takes no questions$"):
        await writer.dispatch("writer", {"input": "x", "questions": {"area": PICK}})
    lead_ = engine.tool("lead", scope="acme")
    assert list(lead_.definitions()[0]["input_schema"]["properties"]) == ["input"]
    assert len(llm.calls) == 1


def _intent(choice: str, p: float) -> dict[str, Answer]:
    rest = (1 - p) / 2
    others = [o for o in ("billing", "technical", "other") if o != choice]
    return {
        "intent": ChoiceAnswer(
            choice=choice, probabilities={choice: p, others[0]: rest, others[1]: rest}
        )
    }


async def test_a_routing_agent_as_a_tool_returns_the_reply_it_routed_to(fixtures: Path) -> None:
    llm = Scripted(text_reply("Support here."))
    engine = Engine(fixtures, models={"strong": llm, "judge": Fixed(_intent("technical", 0.8))})
    result = await engine.tool("frontdesk", scope="acme").dispatch(
        "frontdesk", {"input": "The app crashes"}
    )
    assert isinstance(result, dict) and result["answers"]["intent"]["choice"] == "technical"
    assert result["routed_to"] == {"agent": "support", "reply": "Support here."}


async def test_a_routing_agent_that_did_not_route_returns_its_answers_only(fixtures: Path) -> None:
    engine = Engine(
        fixtures,
        models={"strong": Scripted(), "judge": Fixed(_intent("technical", 0.5))},
    )
    result = await engine.tool("frontdesk", scope="acme").dispatch("frontdesk", {"input": "x"})
    assert isinstance(result, dict) and list(result) == ["answers"]


async def test_a_delegated_routing_agent_hands_back_the_reply_it_routed_to(fixtures: Path) -> None:
    llm = Scripted(text_reply("Support here."))
    engine = Engine(fixtures, models={"strong": llm, "judge": Fixed(_intent("technical", 0.8))})
    run = await engine.run("frontdesk", "acme", "The app crashes")
    handed = json.loads(child_result(run))
    assert handed["answers"]["intent"]["choice"] == "technical"
    assert handed["routed_to"] == {"agent": "support", "reply": "Support here."}
