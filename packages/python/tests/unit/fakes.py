from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from rfnry_engine.models import (
    Answer,
    Check,
    CheckAnswer,
    ChoiceAnswer,
    Decided,
    Message,
    QuestionLike,
    Reply,
    ScoreAnswer,
    State,
    ToolCall,
    ToolSpec,
    Usage,
)
from rfnry_engine.records import Decision, Generation, Run, Step
from rfnry_engine.tools.tools import Tool


class Scripted:
    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls.append((system, list(messages), list(tools)))
        if not self.replies:
            raise AssertionError("Scripted ran out of replies")
        return self.replies.pop(0)


class Streaming(Scripted):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> Reply:
        reply = await self.generate(system, messages, tools)
        for piece in reply.text.split(" "):
            await on_delta(piece + " ")
        return reply


class Fixed:
    def __init__(
        self, answers: dict[str, Answer] | Callable[[State], dict[str, Answer] | Decided]
    ) -> None:
        self.answers = answers
        self.calls: list[tuple[State, dict[str, QuestionLike]]] = []

    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> dict[str, Answer] | Decided:
        self.calls.append((state, dict(questions)))
        if callable(self.answers):
            return self.answers(state)
        return dict(self.answers)


def text_reply(text: str, **kwargs: Any) -> Reply:
    return Reply(text=text, stop_reason="end_turn", **kwargs)


async def lookup_order(args: dict[str, Any]) -> Any:
    return {"order": args["order_id"]}


async def now(args: dict[str, Any]) -> str:
    return "2026-09-22T00:00:00Z"


TOOLS = [Tool("lookup_order", body=lookup_order), Tool("now", "The current UTC time", body=now)]


def generation_record(
    scope: str = "acme",
    agent: str = "support",
    *,
    reply: str = "ok",
    children: list[Run] | None = None,
    run_id: str | None = None,
) -> Generation:
    return Generation(
        run_id=run_id or uuid4().hex,
        agent=agent,
        scope=scope,
        at=datetime.now(UTC),
        duration_ms=5,
        model="m",
        input="hi",
        system="<<<AGENT>>>\nx",
        messages=[Message(role="user", content="hi")],
        conversation=[Message(role="user", content="hi"), Message(role="assistant", content=reply)],
        steps=[
            Step(
                tool="Read",
                args={"path": "a"},
                result="A",
                is_error=False,
                at=datetime.now(UTC),
                duration_ms=1,
            )
        ],
        calls=1,
        reply=reply,
        usage=Usage(input=3, output=1),
        children=list(children or []),
    )


def decision_record(
    scope: str = "acme",
    agent: str = "triage",
    *,
    probability: float = 0.8,
    routed_to: Run | None = None,
    run_id: str | None = None,
) -> Decision:
    return Decision(
        run_id=run_id or uuid4().hex,
        agent=agent,
        scope=scope,
        at=datetime.now(UTC),
        duration_ms=2,
        model="d",
        state={"message": "hi"},
        questions={"urgent": Check(instructions="Urgent?")},
        answers={
            "urgent": CheckAnswer(probability=probability, confidence=abs(2 * probability - 1))
        },
        usage=Usage(input=2),
        routed_to=routed_to,
    )


class MemoryBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data


class Desk:
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> dict[str, Answer]:
        if "intent" in questions:
            return {
                "intent": ChoiceAnswer(
                    choice="technical",
                    probabilities={"technical": 0.9, "billing": 0.05, "other": 0.05},
                )
            }
        return {
            "urgent": CheckAnswer(probability=0.9),
            "severity": ScoreAnswer(score=3, probabilities=[0, 0, 0, 1]),
        }


def delegate_triage(call_id: str = "d1", text: str = "It is down") -> Reply:
    call = ToolCall(id=call_id, name="Delegate", args={"agent": "triage", "input": text})
    return Reply(tool_calls=[call], stop_reason="tool_use")
