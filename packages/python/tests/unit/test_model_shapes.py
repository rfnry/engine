from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import (
    AnswerLike,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Decided,
    DecidedLike,
    DecisionModel,
    GenerationModel,
    Image,
    Message,
    QuestionLike,
    Reply,
    ReplyLike,
    Score,
    ScoreAnswer,
    State,
    StreamingGenerationModel,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
)
from rfnry_engine.records import Decision, Generation

from . import lookalike as other
from .fakes import TOOLS, Fixed, Scripted, Streaming


class Adapter:
    def __init__(self) -> None:
        self.replies = [
            other.Reply(
                tool_calls=[
                    other.ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"})
                ],
                stop_reason="tool_use",
            ),
            other.Reply(text="done", usage=other.Usage(input=3, output=1), model="m"),
        ]

    async def generate(
        self,
        system: str,
        messages: Sequence[Message | other.Message],
        tools: Sequence[ToolSpec | other.ToolSpec],
    ) -> other.Reply:
        return self.replies.pop(0)

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> other.Decided:
        answers: dict[str, other.Answer] = {
            "urgent": other.CheckAnswer(probability=0.9),
            "severity": other.ScoreAnswer(score=1, probabilities=[0, 1, 0, 0]),
        }
        return other.Decided(answers=answers, usage=other.Usage(input=7), model="d")


async def test_an_adapter_typed_against_lookalike_classes_runs_in_the_engine(
    fixtures: Path,
) -> None:
    adapter = Adapter()
    engine = Engine(fixtures, generation=adapter, decision=adapter, tools=TOOLS)
    generation = await engine.run("support", "acme", "x")
    assert isinstance(generation, Generation) and generation.reply == "done"
    assert generation.steps[0].tool == "Read" and generation.usage == Usage(input=3, output=1)
    decision = await engine.run("triage", "acme", "x")
    assert isinstance(decision, Decision) and decision.model == "d"
    assert isinstance(decision.answers["urgent"], CheckAnswer)
    assert decision.answers["urgent"].confidence == pytest.approx(0.8)


def test_the_engine_classes_have_the_shapes_the_engine_reads() -> None:
    reply: ReplyLike = Reply(tool_calls=[ToolCall(id="c1", name="Read")], usage=Usage())
    answers: Mapping[str, AnswerLike] = {
        "a": ChoiceAnswer(choice="x", probabilities={"x": 1.0}),
        "b": ScoreAnswer(score=0, probabilities=[1.0]),
        "c": CheckAnswer(probability=1.0),
    }
    decided: DecidedLike = Decided(answers={"c": CheckAnswer(probability=1.0)})
    generation: GenerationModel = Scripted()
    streaming: StreamingGenerationModel = Streaming()
    decision: DecisionModel = Fixed({})
    assert reply.tool_calls[0].name == "Read" and decided.answers and len(answers) == 3
    assert generation and streaming and decision


class Plain(BaseModel):
    type: Literal["check"] = "check"
    instructions: str
    yes: str | None = None
    no: str | None = None


def test_structured_questions_from_either_library_and_plain_ones_are_question_like() -> None:
    questions: Mapping[str, QuestionLike] = {
        "a": Choice(instructions={"question": "Kind?"}, options={"x": {"what": "X"}, "y": None}),
        "b": Score(instructions=["Same?", {"note": "names"}], levels=[{"summary": "no"}, "yes"]),
        "c": Check(instructions="Relevant?", yes=["on topic"], no={"what": "elsewhere"}),
        "d": other.Choice(instructions={"question": "Kind?"}, options={"x": ["X"], "y": None}),
        "e": other.Score(instructions="Same?", levels=[["no"], {"summary": "yes"}]),
        "f": other.Check(instructions={"question": "Relevant?"}, no="elsewhere"),
        "g": Plain(instructions="Relevant?", yes="on topic"),
    }
    read: dict[str, list[object]] = {}
    for qid, question in questions.items():
        if question.type == "choice":
            read[qid] = [question.instructions, dict(question.options)]
        elif question.type == "score":
            read[qid] = [question.instructions, list(question.levels)]
        else:
            read[qid] = [question.instructions, question.yes, question.no]
    assert read["a"] == [{"question": "Kind?"}, {"x": {"what": "X"}, "y": None}]
    assert read["e"] == ["Same?", [["no"], {"summary": "yes"}]]
    assert read["g"] == ["Relevant?", "on topic", None]


async def test_parts_from_another_library_are_read_by_their_fields(fixtures: Path) -> None:
    llm = Scripted(
        Reply(tool_calls=[ToolCall(id="p", name="photo")], stop_reason="tool_use"),
        Reply(text="A cat."),
    )
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    run = await engine.run(
        "billing",
        "acme",
        [other.Text(text="And this?"), other.Image(media_type="image/png", url="https://x/c.png")],
        tools=other.Photos(),
    )
    assert isinstance(run, Generation)
    assert llm.calls[0][1][-1].content == [
        Text(text="And this?"),
        Image(media_type="image/png", url="https://x/c.png"),
    ]
    assert llm.calls[1][1][-1].content == [
        Text(text="the photo"),
        Image(media_type="image/png", data=b"png"),
    ]


async def test_history_from_another_library_is_read_by_its_fields(fixtures: Path) -> None:
    llm = Scripted(Reply(text="Still a cat."))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    history = [
        other.Message(
            role="user",
            content=[
                other.Text(text="And this?"),
                other.Image(media_type="image/png", url="https://x/c.png"),
            ],
        ),
        other.Message(
            role="assistant", content="A cat.", tool_calls=[other.ToolCall(id="c1", name="Read")]
        ),
        other.Message(role="tool", tool_call_id="c1", content="a cat"),
        {
            "role": "user",
            "content": [other.Text(text="Sure?"), other.Image(media_type="image/png", data=b"png")],
        },
    ]
    run = await engine.run("billing", "acme", "Look again.", history=history)
    assert isinstance(run, Generation)
    sent = llm.calls[0][1]
    assert sent[0] == Message(
        role="user",
        content=[Text(text="And this?"), Image(media_type="image/png", url="https://x/c.png")],
    )
    assert sent[1] == Message(
        role="assistant", content="A cat.", tool_calls=[ToolCall(id="c1", name="Read")]
    )
    assert sent[2] == Message(role="tool", tool_call_id="c1", content="a cat")
    assert sent[3].content == [Text(text="Sure?"), Image(media_type="image/png", data=b"png")]
    assert run.conversation[:4] == sent[:4]


async def test_a_malformed_part_from_another_library_in_history_is_an_input_error(
    fixtures: Path,
) -> None:
    engine = Engine(fixtures, generation=Scripted(Reply(text="x")), tools=TOOLS)
    missing = other.Message.model_construct(
        role="user", content=[other.Text.model_construct(type="text")]
    )
    with pytest.raises(InputError, match=r"^engine: history does not fit the contract"):
        await engine.run("billing", "acme", "x", history=[missing])
    both = {
        "role": "user",
        "content": [other.Image(media_type="image/png", data=b"png", url="https://x/c.png")],
    }
    with pytest.raises(InputError, match=r"^models: image carries exactly one of data or url"):
        await engine.run("billing", "acme", "x", history=[both])
