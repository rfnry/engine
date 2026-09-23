from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, ModelError
from rfnry_engine.models import (
    CheckAnswer,
    ChoiceAnswer,
    Decided,
    Image,
    Message,
    Part,
    Reply,
    ScoreAnswer,
    State,
    Text,
    ToolCall,
    Usage,
)
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Fixed, Scripted, text_reply


def _intent(choice: str, p: float) -> dict[str, Any]:
    rest = (1 - p) / 2
    others = [o for o in ("billing", "technical", "other") if o != choice]
    return {"intent": ChoiceAnswer(choice=choice, probabilities={choice: p, others[0]: rest, others[1]: rest})}


async def test_decision_run_settles_answers(fixtures: Path) -> None:
    dm = Fixed({"urgent": CheckAnswer(probability=0.9), "severity": ScoreAnswer(score=2.4, probabilities=[0.0, 0.1, 0.4, 0.5])})
    run = await Engine(fixtures, decision=dm, tools=TOOLS).run("triage", "acme", {"message": "It is down"})
    assert isinstance(run, Decision)
    assert run.answers["urgent"].confidence == pytest.approx(0.8)
    assert run.answers["severity"].confidence == pytest.approx(1 / 3)
    assert set(run.questions) == {"urgent", "severity"} and run.state == {"message": "It is down"}
    assert dm.calls[0][0] == {"message": "It is down"} and run.routed_to is None


async def test_route_runs_the_target_with_the_same_input(fixtures: Path) -> None:
    llm = Scripted(text_reply("Support here."))
    engine = Engine(fixtures, generation=llm, decision=Fixed(_intent("technical", 0.8)), tools=TOOLS)
    run = await engine.run("frontdesk", "acme", "The app crashes", context="Team plan")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.agent == "support" and run.routed_to.reply == "Support here."
    assert run.routed_to.parent_run_id == run.run_id and run.routed_to.input == "The app crashes"
    assert "<<<CONTEXT>>>\nTeam plan" in run.routed_to.system


async def test_no_route_below_the_floor_or_for_an_unmapped_option(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), decision=Fixed(_intent("technical", 0.5)), tools=TOOLS)
    low = await engine.run("frontdesk", "acme", "x")
    assert isinstance(low, Decision) and low.routed_to is None
    engine = Engine(fixtures, generation=Scripted(), decision=Fixed(_intent("other", 0.99)), tools=TOOLS)
    other = await engine.run("frontdesk", "acme", "x")
    assert isinstance(other, Decision) and other.routed_to is None


async def test_decided_carries_usage_and_model(fixtures: Path) -> None:
    def decide(state: State) -> Any:
        return Decided(answers={"urgent": CheckAnswer(probability=0.2), "severity": ScoreAnswer(score=0, probabilities=[1, 0, 0, 0])}, usage=Usage(input=40), model="d1")

    run = await Engine(fixtures, decision=Fixed(decide), tools=TOOLS).run("triage", "acme", "x")
    assert isinstance(run, Decision) and run.usage == Usage(input=40) and run.model == "d1"


async def test_a_decision_only_engine_runs_a_mixed_tree_without_tool_bodies(fixtures: Path) -> None:
    dm = Fixed({"urgent": CheckAnswer(probability=0.9), "severity": ScoreAnswer(score=1, probabilities=[0, 1, 0, 0])})
    run = await Engine(fixtures, decision=dm).run("triage", "acme", "x")
    assert isinstance(run, Decision)
    with pytest.raises(ConfigurationError, match="weather"):
        Engine(fixtures, decision=dm, tools=[Tool("weather", "Forecast", body=TOOLS[1].body)])


async def test_a_route_to_an_agent_without_its_model_is_refused_before_the_decision(fixtures: Path) -> None:
    dm = Fixed(_intent("other", 0.9))
    message = "^engine: agent 'frontdesk' routes to generation agent 'billing'; no generation model was configured$"
    with pytest.raises(ConfigurationError, match=message):
        await Engine(fixtures, decision=dm).run("frontdesk", "acme", "x")
    assert dm.calls == []


class ForeignUsage(BaseModel):
    input: int = 0
    output: int = 0


class ForeignCheck(BaseModel):
    type: Literal["check"] = "check"
    probability: float
    confidence: float = 0.0


class ForeignScore(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0


class ForeignDecided(BaseModel):
    answers: dict[str, ForeignCheck | ForeignScore]
    usage: ForeignUsage | None = None
    model: str | None = None


def _foreign(probability: float) -> Any:
    return {"urgent": ForeignCheck(probability=probability), "severity": ForeignScore(score=1, probabilities=[0, 1, 0, 0])}


async def test_a_foreign_decided_is_accepted_by_its_fields(fixtures: Path) -> None:
    def decide(state: State) -> Any:
        return ForeignDecided(answers=_foreign(0.9), usage=ForeignUsage(input=7), model="d2")

    run = await Engine(fixtures, decision=Fixed(decide), tools=TOOLS).run("triage", "acme", "x")
    assert isinstance(run, Decision) and run.usage == Usage(input=7) and run.model == "d2"
    assert isinstance(run.answers["urgent"], CheckAnswer) and run.answers["urgent"].confidence == pytest.approx(0.8)
    assert isinstance(run.answers["severity"], ScoreAnswer)


async def test_foreign_answers_alone_are_accepted_and_still_validated(fixtures: Path) -> None:
    run = await Engine(fixtures, decision=Fixed(_foreign(0.9)), tools=TOOLS).run("triage", "acme", "x")
    assert isinstance(run, Decision) and isinstance(run.answers["urgent"], CheckAnswer)
    with pytest.raises(ModelError, match="^models: the decision model's answers do not fit the contract"):
        await Engine(fixtures, decision=Fixed(_foreign(1.5)), tools=TOOLS).run("triage", "acme", "x")


async def test_a_decision_agent_takes_no_task(fixtures: Path) -> None:
    engine = Engine(fixtures, decision=Fixed(_intent("other", 0.9)), tools=TOOLS)
    with pytest.raises(InputError, match="^engine: decision agent 'frontdesk' takes no task$"):
        await engine.run("frontdesk", "acme", "x", task="triage")


class Verdict(BaseModel):
    refund: bool


async def test_a_route_forwards_everything_the_run_was_given(fixtures: Path) -> None:
    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    output = ToolCall(id="o", name="Output", args={"refund": True})
    pinged = Reply(tool_calls=[ToolCall(id="p", name="ping")], stop_reason="tool_use")
    llm = Scripted(pinged, Reply(tool_calls=[output], stop_reason="tool_use"))
    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    engine = Engine(fixtures, generation=llm, decision=Fixed(_intent("technical", 0.9)), tools=TOOLS)
    history = [Message(role="user", content="earlier"), Message(role="assistant", content="noted")]
    run = await engine.run(
        "frontdesk", "acme", "x", history=history, context="Team plan", expect=Verdict,
        on_event=on_event, tools=Tools([Tool("ping", "Ping", body=ping)]),
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    system, messages, specs = llm.calls[0]
    assert messages[:2] == history and "<<<CONTEXT>>>\nTeam plan" in system
    assert {"ping", "Output"} <= {spec.name for spec in specs}
    assert Verdict.model_validate_json(run.routed_to.reply).refund is True
    assert [type(e).__name__ for e in events] == ["ToolCallStarted", "ToolResultReady"]


async def test_a_decision_agent_refuses_media_before_the_model_is_called(fixtures: Path) -> None:
    dm = Fixed({"urgent": CheckAnswer(probability=0.9), "severity": ScoreAnswer(score=1, probabilities=[0, 1, 0, 0])})
    engine = Engine(fixtures, decision=dm, tools=TOOLS)
    parts = [Text(text="Is this down?"), Image(media_type="image/png", url="https://x/screen.png")]
    with pytest.raises(InputError, match="^engine: decision agent 'triage' takes text only; got image$"):
        await engine.run("triage", "acme", parts)
    assert dm.calls == []


async def test_a_decision_agent_joins_text_parts_into_its_state(fixtures: Path) -> None:
    dm = Fixed({"urgent": CheckAnswer(probability=0.9), "severity": ScoreAnswer(score=1, probabilities=[0, 1, 0, 0])})
    run = await Engine(fixtures, decision=dm, tools=TOOLS).run("triage", "acme", [Text(text="It is down."), Text(text="Since noon.")])
    assert isinstance(run, Decision) and run.state == "It is down.\nSince noon."
    assert dm.calls[0][0] == "It is down.\nSince noon."


async def test_a_route_forwards_the_parts_to_its_target(fixtures: Path) -> None:
    llm = Scripted(text_reply("Support here."))
    engine = Engine(fixtures, generation=llm, decision=Fixed(_intent("technical", 0.8)), tools=TOOLS)
    parts: list[Part] = [Text(text="The app crashes"), Text(text="on login")]
    run = await engine.run("frontdesk", "acme", parts)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.state == "The app crashes\non login" and run.routed_to.input == parts
    assert llm.calls[0][1][-1] == Message(role="user", content=parts)
