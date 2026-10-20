import ast
import asyncio
import hashlib
import inspect
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import rfnry_engine
from rfnry_engine import (
    CheckAnswer,
    ChoiceAnswer,
    Document,
    Engine,
    Local,
    Message,
    Part,
    QuestionLike,
    Refinery,
    Reply,
    ScoreAnswer,
    State,
    Storage,
    Text,
    Tool,
    ToolCall,
    Tools,
    ToolSpec,
    Workspace,
)
from rfnry_engine.records import Decision
from rfnry_engine.tree import load_agent, load_tree, validate_tree

from .fakes import lookup_order

DOCS = Path(__file__).parents[4] / "docs" / "engine.md"
BLOCKS: list[tuple[str, str]] = re.findall(
    r"```(\w*)\n(.*?)```", DOCS.read_text(encoding="utf-8"), re.S
)
PYTHON = [body for lang, body in BLOCKS if lang == "python"]
MARKDOWN = [body for lang, body in BLOCKS if lang in ("markdown", "")]


def first_line(block: str) -> str:
    return block.splitlines()[0]


def markdown(start: str) -> str:
    return next(block for block in MARKDOWN if start in block)


ILLUSTRATIVE = {
    "class Conversation:": "needs a database pool; a list stands in for it in the block above",
    "refinery = Refinery(PostgresStorage(pool))": "needs a Postgres pool; the integration tier runs PostgresStorage",
    "pool = await asyncpg.create_pool(dsn)": "needs a Postgres pool and rfnry-memory",
    "from rfnry_engine import Local, RunStarted, Tool, ToolResultReady, Tools, Workspace, current_call": (
        "needs agents that declare `workspace: true`"
    ),
    "jobs = {}": "needs a model that calls start and then check",
}

SIGNATURES = {
    "class GenerationModel(Protocol):": (rfnry_engine.GenerationModel, rfnry_engine.DecisionModel),
    "class Tools:": (Tools,),
    "class Storage(Protocol):": (rfnry_engine.Storage, Refinery),
    "class Backend(Protocol):": (rfnry_engine.Backend,),
}


def delegating(agent: str) -> Reply:
    call = ToolCall(id="d1", name="Delegate", args={"agent": agent, "input": "It is down"})
    return Reply(tool_calls=[call], stop_reason="tool_use")


COMPOSED = json.loads(next(body for lang, body in BLOCKS if lang == "json" and "questions" in body))
ASKS = 'run = await engine.run("support", scope="acme", input="I was charged twice in March. Refund one.")'
ROUTE = next(body for lang, body in BLOCKS if lang == "yaml" and "task:" in body)


@dataclass
class Setup:
    replies: tuple[Reply, ...] = ()
    intent: str = "billing"
    replace: dict[str, str] = field(default_factory=dict)


SETUPS = {
    'engine = Engine(Path("skills"), generation=llm, name="writer")': Setup(
        replace={'Path("skills")': "skills"}
    ),
    "class Budget:": Setup(replies=(delegating("triage"),), intent="technical"),
    "def spent(run):": Setup(replies=(delegating("triage"),), intent="technical"),
    "class Verdict(BaseModel):": Setup(
        replies=(
            Reply(
                tool_calls=[
                    ToolCall(id="o1", name="Output", args={"refund": True, "reason": "dup"})
                ],
                stop_reason="tool_use",
            ),
        )
    ),
    "engine = Engine(root, generation=llm, decision=dm, tools=tools)": Setup(
        replies=(delegating("triage"),), intent="technical"
    ),
    ASKS: Setup(
        replies=(
            Reply(
                tool_calls=[ToolCall(id="d1", name="Delegate", args=COMPOSED)],
                stop_reason="tool_use",
            ),
        )
    ),
    "def forgotten(message):": Setup(
        replace={
            "map(forgotten, run.added)]": (
                "map(forgotten, [*run.added, Message(role='user', content=["
                "Text(text='This one?'), Image(media_type='image/jpeg', url='https://x/a.jpg'), "
                "Image(media_type='image/png', data=png)])])]\n"
                "assert conversation[-1].content[1:] == ["
                "Text(text='[image: image/jpeg, https://x/a.jpg]'), "
                "Image(media_type='image/png', data=png)]\n"
                "assert conversation[:-1] == run.added\n"
                "assert await see({'url': 'https://x/a.jpg', 'media_type': 'image/jpeg'}) == "
                "[Image(media_type='image/jpeg', url='https://x/a.jpg')]"
            )
        }
    ),
    "from rfnry_engine import Local, Workspace": Setup(
        replace={"/srv/tenants/acme/work": "work", '"bwrap",': '"env",'}
    ),
    "async def approve(agent, call):": Setup(
        replies=(
            Reply(
                tool_calls=[ToolCall(id="s1", name="Shell", args={"command": "echo hi"})],
                stop_reason="tool_use",
            ),
        ),
        replace={"/srv/tenants/acme/work": "work"},
    ),
}


class Model:
    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        return self.replies.pop(0) if self.replies else Reply(text="ok")


class Answering:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def answer(self, qid: str, question: QuestionLike) -> Any:
        if question.type == "choice":
            chosen = (
                self.intent if self.intent in question.options else next(iter(question.options))
            )
            rest = (1.0 - 0.9) / (len(question.options) - 1)
            return ChoiceAnswer(
                choice=chosen,
                probabilities={o: 0.9 if o == chosen else rest for o in question.options},
            )
        if question.type == "score":
            rest = (1.0 - 0.9) / (len(question.levels) - 1)
            return ScoreAnswer(score=0, probabilities=[0.9, *[rest] * (len(question.levels) - 1)])
        return CheckAnswer(probability=0.91)

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        return {qid: self.answer(qid, question) for qid, question in questions.items()}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent(name: str, kind: str, body: str = "") -> str:
    return f"---\nkind: {kind}\ndescription: The {name} agent\n---\n{body}\n"


def question(kind: str, extra: str = "") -> str:
    return f"---\ntype: {kind}\n{extra}---\nA question.\n"


def skills_folder(tmp_path: Path) -> Path:
    write(tmp_path / "skills/launch.md", "---\ndescription: The launch checklist\n---\nShip it.\n")
    return tmp_path / "skills"


def tree(root: Path) -> None:
    write(
        root / "support/AGENT.md", markdown("description: Handles technical support conversations")
    )
    write(
        root / "support/rules/refunds.md",
        "---\ndescription: When a refund is owed\n---\nA double charge is refunded in full.\n",
    )
    write(root / "support/tasks/triage.md", "Triage the message.\n")
    write(
        root / "support/skills/escalate.md",
        "---\ndescription: Hand a conversation to a person\n---\nSay so, then stop.\n",
    )
    write(root / "support/TOOLS.md", markdown("fetch_invoice: Fetch"))
    write(
        root / "support/tools/lookup_order.md",
        markdown("description: Fetch an order by id\ninput:"),
    )
    write(root / "frontdesk/AGENT.md", markdown("description: Routes an incoming message"))
    write(
        root / "frontdesk/questions/intent.md",
        markdown("Which department should handle this message?"),
    )
    write(root / "coder/AGENT.md", markdown("description: Writes and runs code").split("\n", 1)[1])
    write(root / "billing/AGENT.md", agent("billing", "generation"))
    write(root / "teacher/AGENT.md", agent("teacher", "generation"))
    write(root / "triage/AGENT.md", agent("triage", "decision"))
    write(root / "triage/questions/urgent.md", markdown("type: check"))
    write(root / "triage/questions/severity.md", markdown("type: score"))
    write(root / "grader/AGENT.md", agent("grader", "decision"))
    write(root / "grader/questions/grounded.md", question("check"))
    write(root / "grader/tasks/tone/polite.md", question("check"))
    write(
        root / "grader/tasks/tone/register.md",
        question("choice", "options: {formal: Formal, casual: Casual}\n"),
    )
    write(root / "judge/AGENT.md", agent("judge", "decision"))
    write(root / "judge/questions/grounded.md", question("check"))
    write(root / "judge/questions/rule.md", question("check"))
    write(
        root / "judge/questions/done.md",
        question("score", "levels: [Not addressed, Partly, Fully, Fully and confirmed]\n"),
    )


class InvoiceArgs(BaseModel):
    invoice_id: str


async def fetch_invoice(args: dict[str, Any]) -> list[Part]:
    return [Document(media_type="application/pdf", name="a.pdf", data=b"%PDF-1.4")]


class Invoices:
    async def get(self, invoice_id: str) -> bytes:
        return b"%PDF-1.4"


TOOLS = [
    Tool("lookup_order", body=lookup_order),
    Tool("fetch_invoice", "Fetch an invoice PDF by id", body=fetch_invoice, args=InvoiceArgs),
]


def memory_tools(memory: object, corpus_id: str) -> Tools:
    async def recall(args: dict[str, Any]) -> str:
        return f"nothing remembered for {corpus_id}"

    return Tools([Tool("recall", "Recall a fact", body=recall)])


async def ask_a_person(agent: str, call: ToolCall) -> bool:
    return call.name == "Shell" and call.args == {"command": "echo hi"}


async def context(tmp_path: Path, setup: Setup, storage: Storage) -> dict[str, Any]:
    root = tmp_path / "agents"
    tree(root)
    (tmp_path / "work").mkdir()
    refinery = Refinery(storage)
    recorded = Engine(
        root, generation=Model(), decision=Answering("billing"), tools=TOOLS, refinery=refinery
    )
    llm, dm = Model(*setup.replies), Answering(setup.intent)
    return {
        **{name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__},
        "Path": Path,
        "BaseModel": BaseModel,
        "asyncio": asyncio,
        "root": root,
        "skills": skills_folder(tmp_path),
        "llm": llm,
        "dm": dm,
        "refinery": refinery,
        "engine": Engine(root, generation=llm, decision=dm, tools=TOOLS, refinery=refinery),
        "tools": list(TOOLS),
        "InvoiceArgs": InvoiceArgs,
        "fetch_invoice": fetch_invoice,
        "invoices": Invoices(),
        "lookup_order": lookup_order,
        "png": b"\x89PNG\r\n\x1a\n",
        "wav": b"RIFF",
        "MemoryTools": memory_tools,
        "memory": object(),
        "message": "My invoice was charged twice.",
        "text": "It is down",
        "history": [],
        "run": await recorded.run("support", "acme", "My invoice was charged twice."),
        "decision": await recorded.run("frontdesk", "acme", "It is down"),
        "last_watermark": None,
        "ask_a_person": ask_a_person,
        "workspace": Workspace(Local(tmp_path / "work"), scope="acme"),
        "hashlib": hashlib,
        "checkout": lambda commit: shutil.copytree(
            root, tmp_path / "versions" / commit, dirs_exist_ok=True
        ),
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


@pytest.mark.parametrize(
    "block", [b for b in PYTHON if first_line(b) in SIGNATURES], ids=first_line
)
def test_signature_blocks_match_the_code(block: str) -> None:
    documented = dict(re.findall(r"def (\w+)\(([^)]*)\)", block))
    classes = SIGNATURES[first_line(block)]
    if getattr(classes[0], "_is_protocol", False):
        members = {n for n, v in vars(classes[0]).items() if inspect.isfunction(v) and n[0] != "_"}
        assert members <= set(documented), sorted(members - set(documented))
    for method, params in documented.items():
        real = next(getattr(c, method) for c in classes if hasattr(c, method))
        while re.search(r"\[[^\[\]]*\]", params):
            params = re.sub(r"\[[^\[\]]*\]", "", params)
        names = [p.split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()]
        expected: list[str] = []
        for p in inspect.signature(real).parameters.values():
            if p.kind is p.KEYWORD_ONLY and not any(e.startswith("*") for e in expected):
                expected.append("*")
            expected.append(("*" if p.kind is p.VAR_POSITIONAL else "") + p.name)
        assert names == expected, method
    for name, listed in re.findall(r"^class (\w+)\(BaseModel\): +(.+)$", block, re.M):
        fields = [field.strip() for field in listed.split(",")]
        assert fields == list(getattr(rfnry_engine, name).model_fields), name


@pytest.mark.parametrize("block", runnable())
async def test_docs_example_runs(
    block: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    monkeypatch.chdir(tmp_path)
    setup = SETUPS.get(first_line(block), Setup())
    for old, new in setup.replace.items():
        block = block.replace(old, new)
    namespace = await context(tmp_path, setup, storage)
    code = compile(block, "engine.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if inspect.iscoroutine(result):
        await result


@pytest.mark.parametrize(
    "block", [b for b in MARKDOWN if b.startswith("---\ntype: ")], ids=lambda b: b.split("\n")[1]
)
def test_every_question_in_the_docs_loads(block: str, tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent("a", "decision"))
    write(tmp_path / "a/questions/q.md", block)
    question = load_agent(tmp_path / "a").questions["q"]
    assert question.instructions and question.type in block.split("\n")[1]


async def test_the_question_the_model_composes_reaches_the_decision_agent(
    tmp_path: Path, storage: Storage
) -> None:
    namespace = await context(tmp_path, SETUPS[ASKS], storage)
    block = next(block for block in PYTHON if first_line(block) == ASKS)
    await eval(compile(block, "engine.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
    child = namespace["child"]
    assert isinstance(child, Decision) and child.runtime == ["refund", "cause"]
    assert list(child.questions) == ["severity", "urgent", "refund", "cause"]
    refund, cause = child.answers["refund"], child.answers["cause"]
    assert isinstance(refund, CheckAnswer) and refund.probability == 0.91
    assert isinstance(cause, ChoiceAnswer) and cause.choice == "duplicate"
    assert namespace["run"].steps[0].args == COMPOSED


def test_the_route_entry_with_a_task_loads(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    tree(root)
    front = markdown("description: Routes an incoming message").split("route:")[0]
    write(root / "frontdesk/AGENT.md", f"{front}{ROUTE}---\n")
    agents = load_tree(root)
    validate_tree(agents)
    route = agents["frontdesk"].route
    assert route is not None and route.to["technical"].task == "triage"


def test_the_agent_that_preloads_its_folder_loads(tmp_path: Path) -> None:
    write(tmp_path / "tagger/AGENT.md", markdown("description: Tags each incoming document"))
    write(tmp_path / "tagger/rules/house.md", "Tag by department.\n")
    spec = load_agent(tmp_path / "tagger")
    assert spec.kind == "generation" and spec.body.startswith("Tag the document.")
    assert spec.preload == [Text(text="<<<rules/house.md>>>\nTag by department.\n")]
