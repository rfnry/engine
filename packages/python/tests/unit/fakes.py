import hashlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from rfnry_engine.engine import Engine
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
from rfnry_engine.records import Decision, Fingerprint, Generation, Run, Step
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


def answer_to(question: QuestionLike) -> Answer:
    if question.type == "choice":
        *rest, last = question.options
        share = 0.2 / len(rest)
        return ChoiceAnswer(choice=last, probabilities={**dict.fromkeys(rest, share), last: 0.8})
    if question.type == "score":
        n = len(question.levels)
        return ScoreAnswer(score=(n - 1) / 2, probabilities=[1 / n] * n)
    return CheckAnswer(probability=0.9)


class Answering:
    def __init__(self) -> None:
        self.calls: list[tuple[State, dict[str, QuestionLike]]] = []

    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> dict[str, Answer]:
        self.calls.append((state, dict(questions)))
        return {qid: answer_to(question) for qid, question in questions.items()}


def by_hand(folder: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (path.relative_to(folder).as_posix(), path)
        for path in folder.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    )
    for name, path in files:
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def text_reply(text: str, **kwargs: Any) -> Reply:
    return Reply(text=text, stop_reason="end_turn", **kwargs)


def said(messages: list[Message]) -> list[Message]:
    return [
        message.model_copy(update={"prefix": None, "usage": None, "model": None})
        for message in messages
    ]


async def lookup_order(args: dict[str, Any]) -> Any:
    return {"order": args["order_id"]}


async def now(args: dict[str, Any]) -> str:
    return "2026-09-22T00:00:00Z"


TOOLS = [Tool("lookup_order", body=lookup_order), Tool("now", body=now)]


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
        fingerprint=Fingerprint(agent="a" * 16, tools="b" * 16, questions=None),
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
        fingerprint=Fingerprint(agent="c" * 16, tools="d" * 16, questions="e" * 16),
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


SORTER = {
    "AGENT.md": "---\nname: sorter\nkind: decision\nmodel: judge\n---\n",
    "questions/team.md": (
        "---\ntype: choice\n"
        "instructions:\n  question: Which team handles `ticket`?\n  ticket: {id: 7}\n"
        "options:\n"
        "  billing:\n    what: Charges\n    not_for: Delivery\n    examples: [charged twice]\n"
        "  orders: [Delivery, Returns]\n"
        "  other:\n"
        "---\n"
    ),
    "questions/scope.md": (
        "---\ntype: score\nlevels:\n  - {summary: One change, signals: [a single fix]}\n"
        "  - [Several]\n  - Many\n---\nHow focused is the change?\n"
    ),
    "questions/creds.md": (
        "---\ntype: check\ninstructions:\n  question: Does `message` ask for a credential?\n"
        "  inspect: message\nyes: {what: Asks for a password, examples: [Reply with your PIN]}\n"
        "---\n"
    ),
}


def sorter(root: Path) -> Fixed:
    for name, text in SORTER.items():
        (root / "sorter" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "sorter" / name).write_text(text, encoding="utf-8")
    return Fixed(
        {
            "team": ChoiceAnswer(
                choice="billing", probabilities={"billing": 0.8, "orders": 0.1, "other": 0.1}
            ),
            "scope": ScoreAnswer(score=1.0, probabilities=[0.1, 0.8, 0.1]),
            "creds": CheckAnswer(probability=0.9),
        }
    )


PICK: dict[str, Any] = {
    "type": "choice",
    "instructions": {"question": "Which area is `message` about?", "message": "Charged twice"},
    "options": {"billing": {"what": "Charges", "examples": ["charged twice"]}, "orders": None},
}

LEAD = {
    "lead/AGENT.md": (
        "---\nname: lead\nkind: generation\nmodel: strong\ndescription: Leads the desk\n"
        "agents: [grader, writer, blank]\n---\nLead the desk.\n"
    ),
    "grader/AGENT.md": "---\nname: grader\nkind: decision\nmodel: judge\ndescription: Grades replies\n---\n",
    "grader/questions/urgent.md": "---\ntype: check\n---\nUrgent?\n",
    "grader/tasks/tone/polite.md": "---\ntype: check\n---\nPolite?\n",
    "writer/AGENT.md": "---\nname: writer\nkind: generation\nmodel: strong\ndescription: Writes replies\n---\nWrite.\n",
    "writer/tasks/apology.md": "Apologise first.\n",
    "blank/AGENT.md": "---\nname: blank\nkind: decision\nmodel: judge\ndescription: Answers what it is asked\n---\n",
}


def lead(root: Path) -> Path:
    for name, text in LEAD.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def delegating(agent: str, text: str = "x", call_id: str = "d1", **extra: Any) -> Reply:
    call = ToolCall(id=call_id, name="Delegate", args={"agent": agent, "input": text, **extra})
    return Reply(tool_calls=[call], stop_reason="tool_use")


async def ask(
    engine: Engine,
    agent: str,
    scope: str,
    input: State,
    questions: Mapping[str, QuestionLike] | None = None,
    **options: Any,
) -> Run:
    return await engine._start(agent, scope, input, questions=questions, **options)
