import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, ModelError, NotFoundError
from rfnry_engine.models import (
    Audio,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Decided,
    Document,
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
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Answering, Fixed, MemoryBucket, Scripted, sorter, text_reply


def _intent(choice: str, p: float) -> dict[str, Any]:
    rest = (1 - p) / 2
    others = [o for o in ("billing", "technical", "other") if o != choice]
    return {
        "intent": ChoiceAnswer(
            choice=choice, probabilities={choice: p, others[0]: rest, others[1]: rest}
        )
    }


async def test_decision_run_settles_answers(fixtures: Path) -> None:
    dm = Fixed(
        {
            "urgent": CheckAnswer(probability=0.9),
            "severity": ScoreAnswer(score=2.4, probabilities=[0.0, 0.1, 0.4, 0.5]),
        }
    )
    run = await Engine(fixtures, decision=dm, tools=TOOLS).run(
        "triage", "acme", {"message": "It is down"}
    )
    assert isinstance(run, Decision)
    assert run.answers["urgent"].confidence == pytest.approx(0.8)
    assert run.answers["severity"].confidence == pytest.approx(1 / 3)
    assert set(run.questions) == {"urgent", "severity"} and run.state == {"message": "It is down"}
    assert dm.calls[0][0] == {"message": "It is down"} and run.routed_to is None


async def test_route_runs_the_target_with_the_same_input(fixtures: Path) -> None:
    llm = Scripted(text_reply("Support here."))
    engine = Engine(
        fixtures, generation=llm, decision=Fixed(_intent("technical", 0.8)), tools=TOOLS
    )
    run = await engine.run("frontdesk", "acme", "The app crashes", context="Team plan")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.agent == "support" and run.routed_to.reply == "Support here."
    assert run.routed_to.parent_run_id == run.run_id and run.routed_to.input == "The app crashes"
    assert "<<<CONTEXT>>>\nTeam plan" in run.routed_to.system


async def test_no_route_below_the_floor_or_for_an_unmapped_option(fixtures: Path) -> None:
    engine = Engine(
        fixtures, generation=Scripted(), decision=Fixed(_intent("technical", 0.5)), tools=TOOLS
    )
    low = await engine.run("frontdesk", "acme", "x")
    assert isinstance(low, Decision) and low.routed_to is None
    engine = Engine(
        fixtures, generation=Scripted(), decision=Fixed(_intent("other", 0.99)), tools=TOOLS
    )
    other = await engine.run("frontdesk", "acme", "x")
    assert isinstance(other, Decision) and other.routed_to is None


async def test_decided_carries_usage_and_model(fixtures: Path) -> None:
    def decide(state: State) -> Any:
        return Decided(
            answers={
                "urgent": CheckAnswer(probability=0.2),
                "severity": ScoreAnswer(score=0, probabilities=[1, 0, 0, 0]),
            },
            usage=Usage(input=40),
            model="d1",
        )

    run = await Engine(fixtures, decision=Fixed(decide), tools=TOOLS).run("triage", "acme", "x")
    assert isinstance(run, Decision) and run.usage == Usage(input=40) and run.model == "d1"


async def test_a_decision_only_engine_runs_a_mixed_tree_without_tool_bodies(fixtures: Path) -> None:
    dm = Fixed(
        {
            "urgent": CheckAnswer(probability=0.9),
            "severity": ScoreAnswer(score=1, probabilities=[0, 1, 0, 0]),
        }
    )
    run = await Engine(fixtures, decision=dm).run("triage", "acme", "x")
    assert isinstance(run, Decision)
    with pytest.raises(ConfigurationError, match="weather"):
        Engine(fixtures, decision=dm, tools=[Tool("weather", "Forecast", body=TOOLS[1].body)])


async def test_a_route_to_an_agent_without_its_model_is_refused_before_the_decision(
    fixtures: Path,
) -> None:
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
    return {
        "urgent": ForeignCheck(probability=probability),
        "severity": ForeignScore(score=1, probabilities=[0, 1, 0, 0]),
    }


async def test_a_foreign_decided_is_accepted_by_its_fields(fixtures: Path) -> None:
    def decide(state: State) -> Any:
        return ForeignDecided(answers=_foreign(0.9), usage=ForeignUsage(input=7), model="d2")

    run = await Engine(fixtures, decision=Fixed(decide), tools=TOOLS).run("triage", "acme", "x")
    assert isinstance(run, Decision) and run.usage == Usage(input=7) and run.model == "d2"
    assert isinstance(run.answers["urgent"], CheckAnswer) and run.answers[
        "urgent"
    ].confidence == pytest.approx(0.8)
    assert isinstance(run.answers["severity"], ScoreAnswer)


async def test_foreign_answers_alone_are_accepted_and_still_validated(fixtures: Path) -> None:
    run = await Engine(fixtures, decision=Fixed(_foreign(0.9)), tools=TOOLS).run(
        "triage", "acme", "x"
    )
    assert isinstance(run, Decision) and isinstance(run.answers["urgent"], CheckAnswer)
    with pytest.raises(
        ModelError, match="^models: the decision model's answers do not fit the contract"
    ):
        await Engine(fixtures, decision=Fixed(_foreign(1.5)), tools=TOOLS).run(
            "triage", "acme", "x"
        )


async def test_a_routing_agent_has_no_tasks_so_a_task_run_of_it_is_refused(fixtures: Path) -> None:
    dm = Fixed(_intent("other", 0.9))
    engine = Engine(fixtures, generation=Scripted(), decision=dm, tools=TOOLS)
    with pytest.raises(NotFoundError, match="^engine: agent 'frontdesk' has no task 'triage'$"):
        await engine.run("frontdesk", "acme", "x", task="triage")
    assert dm.calls == []


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

    engine = Engine(
        fixtures, generation=llm, decision=Fixed(_intent("technical", 0.9)), tools=TOOLS
    )
    history = [Message(role="user", content="earlier"), Message(role="assistant", content="noted")]
    run = await engine.run(
        "frontdesk",
        "acme",
        "x",
        history=history,
        context="Team plan",
        expect=Verdict,
        on_event=on_event,
        tools=Tools([Tool("ping", "Ping", body=ping)]),
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    system, messages, specs = llm.calls[0]
    assert messages[:2] == history and "<<<CONTEXT>>>\nTeam plan" in system
    assert {"ping", "Output"} <= {spec.name for spec in specs}
    assert Verdict.model_validate_json(run.routed_to.reply).refund is True
    assert [type(e).__name__ for e in events] == ["ToolCallStarted", "ToolResultReady"]


async def test_a_decision_agent_refuses_media_before_the_model_is_called(fixtures: Path) -> None:
    dm = Fixed(
        {
            "urgent": CheckAnswer(probability=0.9),
            "severity": ScoreAnswer(score=1, probabilities=[0, 1, 0, 0]),
        }
    )
    engine = Engine(fixtures, decision=dm, tools=TOOLS)
    parts = [Text(text="Is this down?"), Image(media_type="image/png", url="https://x/screen.png")]
    with pytest.raises(
        InputError, match="^engine: decision agent 'triage' takes text only; got image/png$"
    ):
        await engine.run("triage", "acme", parts)
    assert dm.calls == []


async def test_a_decision_agent_joins_text_parts_into_its_state(fixtures: Path) -> None:
    dm = Fixed(
        {
            "urgent": CheckAnswer(probability=0.9),
            "severity": ScoreAnswer(score=1, probabilities=[0, 1, 0, 0]),
        }
    )
    run = await Engine(fixtures, decision=dm, tools=TOOLS).run(
        "triage", "acme", [Text(text="It is down."), Text(text="Since noon.")]
    )
    assert isinstance(run, Decision) and run.state == "It is down.\nSince noon."
    assert dm.calls[0][0] == "It is down.\nSince noon."


async def test_a_route_forwards_the_parts_to_its_target(fixtures: Path) -> None:
    llm = Scripted(text_reply("Support here."))
    engine = Engine(
        fixtures, generation=llm, decision=Fixed(_intent("technical", 0.8)), tools=TOOLS
    )
    parts: list[Part] = [Text(text="The app crashes"), Text(text="on login")]
    run = await engine.run("frontdesk", "acme", parts)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.state == "The app crashes\non login" and run.routed_to.input == parts
    assert llm.calls[0][1][-1] == Message(role="user", content=parts)


async def test_a_checks_yes_and_no_reach_the_decision_model(tmp_path: Path) -> None:
    (tmp_path / "triage" / "questions").mkdir(parents=True)
    (tmp_path / "triage" / "AGENT.md").write_text("---\nname: triage\nkind: decision\n---\n")
    (tmp_path / "triage" / "questions" / "urgent.md").write_text(
        "---\n"
        "type: check\n"
        "yes: The service is down or data is at risk\n"
        "no: A question, a request, or a problem with a workaround\n"
        "---\n"
        "Does the message need a person right now?\n"
    )
    dm = Fixed({"urgent": CheckAnswer(probability=0.9)})
    await Engine(tmp_path, decision=dm).run("triage", "acme", "It is down")
    [(_, questions)] = dm.calls
    assert questions["urgent"] == Check(
        instructions="Does the message need a person right now?",
        yes="The service is down or data is at risk",
        no="A question, a request, or a problem with a workaround",
    )


async def test_structured_questions_reach_the_model_the_record_and_the_export_as_written(
    tmp_path: Path,
) -> None:
    dm = sorter(tmp_path)
    refinery = Refinery(InMemoryStorage())
    run = await Engine(tmp_path, decision=dm, refinery=refinery).run("sorter", "acme", "Hi")
    [(_, asked)] = dm.calls
    assert isinstance(run, Decision) and run.questions == asked
    assert run.questions["team"] == Choice(
        instructions={"question": "Which team handles `ticket`?", "ticket": {"id": 7}},
        options={
            "billing": {"what": "Charges", "not_for": "Delivery", "examples": ["charged twice"]},
            "orders": ["Delivery", "Returns"],
            "other": None,
        },
    )
    assert run.questions["creds"] == Check(
        instructions={"question": "Does `message` ask for a credential?", "inspect": "message"},
        yes={"what": "Asks for a password", "examples": ["Reply with your PIN"]},
    )
    assert await refinery.run("acme", run.run_id) == run
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [key] = [k for k in bucket.objects if "/decision/" in k]
    rows = {
        row["question_id"]: row
        for row in map(json.loads, bucket.objects[key].decode().splitlines())
    }
    assert {qid: row["question"] for qid, row in rows.items()} == {
        qid: question.model_dump(mode="json") for qid, question in run.questions.items()
    }
    assert rows["team"]["question"]["options"]["other"] is None
    assert rows["scope"]["question"]["levels"][0] == {
        "summary": "One change",
        "signals": ["a single fix"],
    }


class Seeing(Answering):
    media_types = {"image/*", "audio/wav"}


async def test_media_the_decision_model_declares_is_passed_as_the_state_as_parts(
    tmp_path: Path,
) -> None:
    sorter(tmp_path)
    seeing = Seeing()
    refinery = Refinery(InMemoryStorage())
    engine = Engine(tmp_path, decision=seeing, refinery=refinery)
    parts: list[Part] = [
        Text(text="Which team handles this screenshot?"),
        Image(media_type="image/png", data=b"\x89PNG\r\n\x1a\n"),
        Audio(media_type="audio/wav", url="https://x/call.wav"),
    ]
    run = await engine.run("sorter", "acme", parts)
    [(state, _)] = seeing.calls
    assert state == parts and isinstance(run, Decision) and run.state == parts
    read = await refinery.run("acme", run.run_id)
    assert read == run and isinstance(read, Decision) and read.state == parts
    assert isinstance(read.state, list) and isinstance(read.state[1], Image)
    joined = await engine.run("sorter", "acme", [Text(text="Plain"), Text(text="text")])
    assert isinstance(joined, Decision) and joined.state == "Plain\ntext"


async def test_a_list_state_without_media_reads_back_as_it_was_given(tmp_path: Path) -> None:
    sorter(tmp_path)
    refinery = Refinery(InMemoryStorage())
    engine = Engine(tmp_path, decision=Seeing(), refinery=refinery)
    given: list[Any] = [{"type": "text", "text": "a dict, not a part"}, 1]
    shaped: list[Any] = [{"type": "text", "text": "a dict shaped like a part"}]
    for state in (given, shaped):
        run = await engine.run("sorter", "acme", state)
        read = await refinery.run("acme", run.run_id)
        assert isinstance(read, Decision) and read.state == state
        assert isinstance(read.state, list) and isinstance(read.state[0], dict)


@pytest.mark.parametrize(
    ("model", "part", "message"),
    [
        (
            Answering(),
            Image(media_type="image/png", url="https://x/a.png"),
            "^engine: decision agent 'sorter' takes text only; got image/png$",
        ),
        (
            Seeing(),
            Document(media_type="application/pdf", url="https://x/a.pdf"),
            "^engine: decision agent 'sorter' takes text and audio/wav, image/\\*; got application/pdf$",
        ),
        (
            Seeing(),
            Audio(media_type="audio/mpeg", url="https://x/a.mp3"),
            "^engine: decision agent 'sorter' takes text and audio/wav, image/\\*; got audio/mpeg$",
        ),
    ],
)
async def test_media_the_decision_model_does_not_declare_is_refused_naming_what_it_takes(
    tmp_path: Path, model: Answering, part: Part, message: str
) -> None:
    sorter(tmp_path)
    with pytest.raises(InputError, match=message):
        await Engine(tmp_path, decision=model).run("sorter", "acme", [Text(text="x"), part])
    assert model.calls == []
