import ast
import asyncio
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import rfnry_engine
from rfnry_engine import (
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Document,
    Engine,
    InMemoryStorage,
    Message,
    Part,
    Question,
    Refinery,
    Reply,
    Score,
    ScoreAnswer,
    State,
    Tool,
    ToolCall,
    Tools,
    ToolSpec,
)

from .fakes import lookup_order

DOCS = Path(__file__).parents[4] / "docs" / "engine.md"
BLOCKS: list[tuple[str, str]] = re.findall(r"```(\w*)\n(.*?)```", DOCS.read_text(encoding="utf-8"), re.S)
PYTHON = [body for lang, body in BLOCKS if lang == "python"]
MARKDOWN = [body for lang, body in BLOCKS if lang in ("markdown", "")]


def first_line(block: str) -> str:
    return block.splitlines()[0]


def markdown(start: str) -> str:
    return next(block for block in MARKDOWN if start in block)


ILLUSTRATIVE = {
    "refinery = Refinery(PostgresStorage(pool))": "needs a Postgres pool; the integration tier runs PostgresStorage",
    "pool = await asyncpg.create_pool(dsn)": "needs a Postgres pool and rfnry-memory",
}

SIGNATURES = {
    "class GenerationModel(Protocol):": (rfnry_engine.GenerationModel, rfnry_engine.DecisionModel),
    "class Tools:": (Tools,),
}


def delegating(agent: str) -> Reply:
    call = ToolCall(id="d1", name="Delegate", args={"agent": agent, "input": "It is down"})
    return Reply(tool_calls=[call], stop_reason="tool_use")


@dataclass
class Setup:
    replies: tuple[Reply, ...] = ()
    intent: str = "billing"
    replace: dict[str, str] = field(default_factory=dict)


SETUPS = {
    "class Verdict(BaseModel):": Setup(
        replies=(Reply(tool_calls=[ToolCall(id="o1", name="Output", args={"refund": True, "reason": "dup"})], stop_reason="tool_use"),)
    ),
    "engine = Engine(root, generation=llm, decision=dm, tools=tools)": Setup(replies=(delegating("triage"),), intent="technical"),
    "from rfnry_engine import Workspace": Setup(replace={"/srv/tenants/acme/work": "work"}),
}


class Model:
    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        return self.replies.pop(0) if self.replies else Reply(text="ok")


class Answering:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def answer(self, qid: str, question: Question) -> Any:
        if isinstance(question, Choice):
            chosen = self.intent if self.intent in question.options else next(iter(question.options))
            rest = (1.0 - 0.9) / (len(question.options) - 1)
            return ChoiceAnswer(choice=chosen, probabilities={o: 0.9 if o == chosen else rest for o in question.options})
        if isinstance(question, Score):
            rest = (1.0 - 0.9) / (len(question.levels) - 1)
            return ScoreAnswer(score=0, probabilities=[0.9, *[rest] * (len(question.levels) - 1)])
        return CheckAnswer(probability=0.91)

    async def decide(self, state: State, questions: dict[str, Question]) -> dict[str, Any]:
        return {qid: self.answer(qid, question) for qid, question in questions.items()}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent(name: str, kind: str, body: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\ndescription: The {name} agent\n---\n{body}\n"


def question(kind: str, extra: str = "") -> str:
    return f"---\ntype: {kind}\n{extra}---\nA question.\n"


def tree(root: Path) -> None:
    write(root / "support/AGENT.md", markdown("name: support"))
    write(root / "support/INDEX.md", "rules/refunds.md\n")
    write(root / "support/rules/_index.md", "rules/refunds.md: refunds\n")
    write(root / "support/rules/refunds.md", "A double charge is refunded in full.\n")
    write(root / "support/tasks/triage.md", "Triage the message.\n")
    for name in ("lookup_order", "read_pdf"):
        write(root / f"support/tools/{name}.md", markdown(f"name: {name}\n"))
    write(root / "frontdesk/AGENT.md", markdown("name: frontdesk"))
    write(root / "frontdesk/questions/intent.md", markdown("type: choice"))
    write(root / "coder/AGENT.md", markdown("name: coder").split("\n", 1)[1])
    write(root / "billing/AGENT.md", agent("billing", "generation"))
    write(root / "teacher/AGENT.md", agent("teacher", "generation"))
    write(root / "triage/AGENT.md", agent("triage", "decision"))
    write(root / "triage/questions/urgent.md", markdown("type: check"))
    write(root / "triage/questions/severity.md", markdown("type: score"))
    write(root / "judge/AGENT.md", agent("judge", "decision"))
    write(root / "judge/questions/grounded.md", question("check"))
    write(root / "judge/questions/rule.md", question("check"))
    write(root / "judge/questions/done.md", question("score", "levels: [Not addressed, Partly, Fully, Fully and confirmed]\n"))


class PdfArgs(BaseModel):
    path: str


async def read_pdf(args: dict[str, Any]) -> list[Part]:
    return [Document(media_type="application/pdf", name="a.pdf", data=b"%PDF-1.4")]


TOOLS = [
    Tool("lookup_order", body=lookup_order),
    Tool("read_pdf", "Read a PDF in the agent directory", body=read_pdf, args=PdfArgs),
]


def memory_tools(memory: object, corpus_id: str) -> Tools:
    async def recall(args: dict[str, Any]) -> str:
        return f"nothing remembered for {corpus_id}"

    return Tools([Tool("recall", "Recall a fact", body=recall)])


async def context(tmp_path: Path, setup: Setup) -> dict[str, Any]:
    root = tmp_path / "agents"
    tree(root)
    (tmp_path / "work").mkdir()
    refinery = Refinery(InMemoryStorage())
    recorded = Engine(root, generation=Model(), decision=Answering("billing"), tools=TOOLS, refinery=refinery)
    llm, dm = Model(*setup.replies), Answering(setup.intent)
    return {
        **{name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__},
        "Path": Path, "BaseModel": BaseModel, "asyncio": asyncio,
        "root": root, "llm": llm, "dm": dm, "refinery": refinery,
        "engine": Engine(root, generation=llm, decision=dm, tools=TOOLS, refinery=refinery),
        "tools": list(TOOLS), "PdfArgs": PdfArgs, "read_pdf": read_pdf,
        "lookup_order": lookup_order, "png": b"\x89PNG\r\n\x1a\n", "wav": b"RIFF",
        "MemoryTools": memory_tools, "memory": object(),
        "message": "My invoice was charged twice.", "text": "It is down", "history": [],
        "run": await recorded.run("support", "acme", "My invoice was charged twice."),
        "decision": await recorded.run("frontdesk", "acme", "It is down"),
        "last_watermark": None,
    }


def runnable() -> list[Any]:
    return [
        pytest.param(block, id=first_line(block)[:60])
        for block in PYTHON
        if first_line(block) not in ILLUSTRATIVE and first_line(block) not in SIGNATURES
    ]


def test_every_named_block_is_in_the_docs() -> None:
    lines = [first_line(block) for block in PYTHON]
    assert len(lines) == len(set(lines))
    assert set(ILLUSTRATIVE) | set(SIGNATURES) | set(SETUPS) <= set(lines)


@pytest.mark.parametrize("block", [b for b in PYTHON if first_line(b) in SIGNATURES], ids=first_line)
def test_signature_blocks_match_the_code(block: str) -> None:
    documented = dict(re.findall(r"def (\w+)\(([^)]*)\)", block))
    classes = SIGNATURES[first_line(block)]
    for method, params in documented.items():
        real = next(getattr(c, method) for c in classes if hasattr(c, method))
        while re.search(r"\[[^\[\]]*\]", params):
            params = re.sub(r"\[[^\[\]]*\]", "", params)
        names = [p.split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()]
        expected = [
            ("*" if p.kind is p.VAR_POSITIONAL else "") + p.name for p in inspect.signature(real).parameters.values()
        ]
        assert names == expected, method


@pytest.mark.parametrize("block", runnable())
async def test_docs_example_runs(block: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    setup = SETUPS.get(first_line(block), Setup())
    for old, new in setup.replace.items():
        block = block.replace(old, new)
    namespace = await context(tmp_path, setup)
    code = compile(block, "engine.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if inspect.iscoroutine(result):
        await result
