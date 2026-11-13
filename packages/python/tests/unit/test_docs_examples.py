import ast
import asyncio
import hashlib
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, Field

import rfnry_engine
from rfnry_engine import (
    CheckAnswer,
    ChoiceAnswer,
    Engine,
    Local,
    Message,
    QuestionLike,
    Refinery,
    Reply,
    ScoreAnswer,
    SpokenLine,
    State,
    Storage,
    Tool,
    ToolCall,
    Tools,
    ToolSpec,
    Workspace,
)
from rfnry_engine.compiled import load_folder
from rfnry_engine.manifest import MANIFEST
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools

from .trees import amend, picked_by, provided, runtime, write_folder

DOCS = Path(__file__).parents[4] / "docs" / "engine.md"
BLOCKS: list[tuple[str, str]] = re.findall(
    r"```(\w*)\n(.*?)```", DOCS.read_text(encoding="utf-8"), re.S
)
PYTHON = [body for lang, body in BLOCKS if lang == "python"]
YAML = [body for lang, body in BLOCKS if lang == "yaml"]
LOOKUP = next(body for lang, body in BLOCKS if lang == "json" and "Fetch an order" in body)
SCRIPT = next(body for body in PYTHON if body.startswith("import json\n"))
REFUND_SCHEMA = next(
    body for lang, body in BLOCKS if lang == "json" and "Refund one charge" in body
)
REFUND = next(body for body in PYTHON if body.startswith("import payments\n"))
CLI = next(body for body in PYTHON if body.startswith("import argparse\n"))
CONFTEST = next(body for body in PYTHON if "def check():" in body.splitlines())
REFUND_TEST = next(
    body for body in PYTHON if body.startswith("from types import SimpleNamespace\n")
)
DECLARED = next(body for body in YAML if "module: billing/tools/refund.py" in body)
KINDS = ("choice", "score", "check")
QUESTIONS = [
    body for lang, body in BLOCKS if lang == "json" and json.loads(body).get("type") in KINDS
]
ORDER = next(body for body in QUESTIONS if "customer's orders" in body)
FILLS = next(body for body in PYTHON if body.startswith("from json import load\n"))
CLIENT = next(body for body in PYTHON if body.startswith("import json, os, socket\n"))
MATCHES = 'async with Engine(Path("agents"), providers=providers) as engine:'
ORDERS = [
    {"id": "A-1001", "summary": "Standing desk, shipped"},
    {"id": "A-1002", "summary": "Lamp"},
]


def first_line(block: str) -> str:
    return block.splitlines()[0]


def illustrative(block: str) -> bool:
    return any(line in ILLUSTRATIVE for line in block.splitlines())


def asked(start: str) -> str:
    return next(block for block in QUESTIONS if start in block)


ILLUSTRATIVE = {
    "class Conversation:": "needs a database pool; a list stands in for it in the block above",
    "refinery = Refinery(PostgresStorage(pool))": "needs a Postgres pool; the integration tier runs PostgresStorage",
    'memory_pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "memory, public"})': "needs a Postgres pool and rfnry-memory",
    'pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "engine"})': "needs a Postgres pool; the integration tier runs the command and check()",
    "class PerRun:": "needs agents that declare `workspace: true`",
    "jobs = {}": "needs a model that calls start and then check",
    'ORDERS = os.environ["ORDERS_URL"]': "a tool script, which the test tree holds as support/tools/lookup_order.py",
    "async def run(args, call):": "a tool script, which test_the_refund_tool_runs_in_a_worker_from_a_shell_and_under_its_tests runs",
    "import argparse": "_cli.py, which the refund tool's footer runs from a shell",
    "def check():": "a conftest.py beside the tools, which pytest runs with refund_test.py",
    "from types import SimpleNamespace": "refund_test.py, which pytest runs beside the refund tool",
    "from json import load": "a question's script, which the test tree holds as matcher/questions/order.py",
    "import json, os, socket": "a script a workspace command runs, which the coder's Shell runs as solve.py",
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


@dataclass
class Setup:
    replies: tuple[Reply, ...] = ()
    intent: str = "billing"
    replace: dict[str, str] = field(default_factory=dict)
    amend: dict[str, dict[str, Any]] = field(default_factory=dict)


MODELS = {
    "strong": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "high"},
    "fast": {"provider": "anthropic", "model": "claude-sonnet-5-5"},
    "judge": {"provider": "classifier"},
    "voice-live": {"provider": "live"},
}

SETUPS = {
    "import os": Setup(
        replace={
            '"live": live})': (
                '"live": live})\nrun = await engine.run("drafter", "acme", "Draft it")\n'
                'assert list(engine.models) == ["strong", "fast", "judge", "voice-live"]\n'
                'assert run.model == "claude-sonnet-5-5" and anthropic.cache_info().currsize == 1'
            )
        }
    ),
    "engine = Engine(root, providers=providers)": Setup(
        replace={
            'models={"fast": fast_llm})': (
                'models={"fast": fast_llm})\n'
                "assert run.model == 'fast-1' and list(engine.models) == list(MODELS)"
            )
        }
    ),
    "async def session(start: VoiceStart) -> Spoken:": Setup(
        replace={
            "approve=gate)": (
                "approve=gate)\nassert run.mode == 'duplex' and run.model == 'live-1'\n"
                "assert 'to be spoken' in llm.systems[-1] and 'desk' not in llm.systems[-1]\n"
                "assert [s.tool for s in run.steps] == ['consult'] and run.steps[0].approved\n"
                "assert [line.text for line in run.transcript] == ['Was I charged twice?', 'ok']"
            )
        }
    ),
    "class InvoiceArgs(BaseModel):": Setup(
        amend={"support": {"tools": {"allow": ["lookup_order", "fetch_invoice"]}}},
        replace={
            "tools=invoicing)": (
                "tools=invoicing)\nassert 'fetch_invoice' in [tool.name for tool in run.tools]"
            )
        },
    ),
    "class Budget:": Setup(
        replies=(delegating("triage"),),
        intent="technical",
        replace={
            "Budget(50)": "Budget(0)",
            "stop=budget.stop)": (
                "stop=budget.stop)\nassert run.stopped == 'after_step' and run.calls == 0"
            ),
        },
    ),
    "def spent(run):": Setup(
        replies=(delegating("triage"),),
        intent="technical",
        replace={
            "# calls and tokens the team adds": (
                "# calls and tokens the team adds\nassert more == (2, 0)"
            )
        },
    ),
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
    'engine = Engine(Path("agents"), providers=providers)': Setup(
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
    'run = await engine.run("support", scope="acme", input="A third complaint this week.", skill="escalate")': Setup(
        replace={
            'run.input                                            # "A third complaint."': (
                'assert run.skill == "escalate" and run.input == "A third complaint."'
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
        self.systems: list[str] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.systems.append(system)
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


class Messages:
    def __init__(self, key: str, *, model: str | None, effort: str | None) -> None:
        self.key, self.model, self.effort = key, model, effort

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        return Reply(text="drafted", model=self.model)


class Live:
    model = "live-1"

    async def open(self, *, tools: list[ToolSpec], instructions: str | None) -> object:
        return object()


class Classifier(Answering):
    def __init__(self, key: str, *, model: str | None) -> None:
        super().__init__("billing")


class Realtime(Live):
    def __init__(self, key: str, *, model: str | None) -> None:
        self.key = key


ADAPTERS = types.ModuleType("adapters")
vars(ADAPTERS).update(Messages=Messages, Classifier=Classifier, Realtime=Realtime)


class Session:
    def __init__(self, transport: object, *, live: object, instructions: str) -> None:
        self.live, self.instructions = live, instructions

    async def start(self, tools: Tools) -> None:
        self.heard = await tools.dispatch("consult", {"input": "Was I charged twice?"})


async def until_it_ends(session: Session) -> list[SpokenLine]:
    return [
        SpokenLine(speaker="person", text="Was I charged twice?"),
        SpokenLine(speaker="assistant", text=str(session.heard)),
    ]


async def gate(agent: str, call: ToolCall) -> bool:
    return True


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def question(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, "instructions": "A question.", **fields})


def generation(name: str, model: str = "strong", **entry: Any) -> dict[str, Any]:
    return {
        "kind": "generation",
        "model": model,
        "description": f"The {name} agent",
        "instructions": f"{name}/instructions.md",
        **entry,
    }


def decision(name: str, **entry: Any) -> dict[str, Any]:
    return {
        "kind": "decision",
        "model": "judge",
        "description": f"The {name} agent",
        "instructions": f"{name}/instructions.md",
        "questions": [f"{name}/*.json"],
        **entry,
    }


AGENTS: dict[str, dict[str, Any]] = {
    "support": generation(
        "support",
        description="Handles technical support conversations",
        files={"allow": ["support/knowledge"]},
        skills={"allow": ["support/skills/*"], "public": ["escalate"]},
        tools={"allow": ["lookup_order"]},
        orchestration={"delegate_to": ["triage"]},
        modes={"voice": {"instructions": "support/voice.md"}},
    ),
    "frontdesk": decision(
        "frontdesk",
        description="Routes an incoming message",
        orchestration={
            "route": {
                "by": "intent",
                "to": {"billing": "billing", "technical": "support"},
                "min_confidence": 0.6,
            }
        },
    ),
    "coder": generation(
        "coder", description="Writes and runs code to finish the work.", workspace=True
    ),
    "billing": generation("billing"),
    "teacher": generation("teacher"),
    "drafter": generation("drafter", model="fast"),
    "triage": decision("triage"),
    "matcher": decision("matcher"),
    "judge": decision("judge"),
    "voice": {
        "kind": "voice",
        "model": "voice-live",
        "description": "Talks with a person and consults another agent for facts",
        "instructions": "voice/instructions.md",
        "tools": {"allow": ["consult"]},
        "modes": {"duplex": {"instructions": "voice/instructions_duplex.md"}},
    },
}


def tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_folder(
        root,
        AGENTS,
        models=MODELS,
        runtime_tools=["consult"],
        tools={
            "lookup_order": {
                "module": "support/tools/lookup_order.py",
                "schema": "support/tools/lookup_order.json",
            }
        },
        files={
            **{f"{name}/instructions.md": "" for name in AGENTS},
            "voice/instructions_duplex.md": "You run your own turns.\n",
            "support/instructions.md": "You are the support desk for Acme. Be brief, be exact, cite the rule you are applying.\n",
            "support/voice.md": "Answer in one short sentence, to be spoken.\n",
            "support/knowledge/refunds.md": "---\ndescription: When a refund is owed\n---\nA double charge is refunded in full.\n",
            "support/tools/lookup_order.json": LOOKUP,
            "support/tools/lookup_order.py": SCRIPT,
            "support/skills/escalate/SKILL.md": "---\ndescription: Hand a conversation to a person\n---\nSay so, then stop.\n",
            "frontdesk/intent.json": asked("Which department should handle this message?"),
            "triage/urgent.json": asked("Does the message need a person right now?"),
            "triage/severity.json": asked("How severe is the problem?"),
            "matcher/order.json": ORDER,
            "matcher/order.py": FILLS,
            "judge/grounded.json": question("check"),
            "judge/rule.json": question("check"),
            "judge/done.json": question(
                "score", levels=["Not addressed", "Partly", "Fully", "Fully and confirmed"]
            ),
        },
    )


class Invoices:
    async def get(self, invoice_id: str) -> bytes:
        return b"%PDF-1.4"


def memory_tools(memory: object, corpus_id: str) -> Tools:
    async def recall(args: dict[str, Any]) -> str:
        return f"nothing remembered for {corpus_id}"

    return Tools([Tool("recall", "Recall a fact", body=recall)])


async def ask_a_person(agent: str, call: ToolCall) -> bool:
    return call.name == "Shell" and call.args == {"command": "echo hi"}


def providing(strong: Any, fast: Any, judge: Any, live: Any) -> dict[str, Any]:
    def anthropic(model: str | None, effort: str | None, /) -> Any:
        return fast if model == MODELS["fast"]["model"] else strong

    return {"anthropic": anthropic, **provided({"classifier": judge, "live": live})}


async def context(tmp_path: Path, setup: Setup, storage: Storage) -> dict[str, Any]:
    root = tmp_path / "agents"
    tree(root)
    write(tmp_path / "orders/acme/customers/c-17/orders", json.dumps(ORDERS))
    (tmp_path / "work").mkdir()
    refinery = Refinery(storage)
    recorded = Engine(
        root,
        providers=providing(Model(), Model(), Answering("billing"), Live()),
        refinery=refinery,
    )
    run = await recorded.run("support", "acme", "My invoice was charged twice.")
    decided = await recorded.run("frontdesk", "acme", "It is down")
    for agent, sections in setup.amend.items():
        amend(root, agent, **sections)
    llm, dm, live_model = Model(*setup.replies), Answering(setup.intent), Live()
    fast_llm = Model(Reply(text="drafted", stop_reason="end_turn", model="fast-1"))
    providers = providing(llm, fast_llm, dm, live_model)
    return {
        **{name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__},
        "Path": Path,
        "BaseModel": BaseModel,
        "Field": Field,
        "asyncio": asyncio,
        "MODELS": MODELS,
        "root": root,
        "llm": llm,
        "strong_llm": llm,
        "fast_llm": fast_llm,
        "dm": dm,
        "live_model": live_model,
        "providers": providers,
        "classifier": providers["classifier"],
        "live": providers["live"],
        "Session": Session,
        "transport": object(),
        "until_it_ends": until_it_ends,
        "gate": gate,
        "refinery": refinery,
        "engine": Engine(root, providers=providers, refinery=refinery),
        "invoices": Invoices(),
        "png": b"\x89PNG\r\n\x1a\n",
        "wav": b"RIFF",
        "MemoryTools": memory_tools,
        "memory": object(),
        "message": "My invoice was charged twice.",
        "text": "It is down",
        "history": [],
        "run": run,
        "decision": decided,
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
        if not illustrative(block) and first_line(block) not in SIGNATURES
    ]


def test_every_named_block_is_in_the_docs() -> None:
    lines = [first_line(block) for block in PYTHON]
    assert len(lines) == len(set(lines))
    assert set(SIGNATURES) | set(SETUPS) <= set(lines)


def test_each_illustrative_line_marks_exactly_one_block() -> None:
    marked = {line: sum(line in block.splitlines() for block in PYTHON) for line in ILLUSTRATIVE}
    assert marked == dict.fromkeys(ILLUSTRATIVE, 1)


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
    monkeypatch.setenv("ORDERS_URL", (tmp_path / "orders").as_uri())
    monkeypatch.setitem(sys.modules, "adapters", ADAPTERS)
    for key in ("ANTHROPIC_API_KEY", "CLASSIFIER_API_KEY", "REALTIME_API_KEY"):
        monkeypatch.setenv(key, "a key of the test's")
    setup = SETUPS.get(first_line(block), Setup())
    for old, new in setup.replace.items():
        block = block.replace(old, new)
    namespace = await context(tmp_path, setup, storage)
    code = compile(block, "engine.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if inspect.iscoroutine(result):
        await result


@pytest.mark.parametrize("block", QUESTIONS, ids=lambda b: json.loads(b)["type"])
def test_every_question_in_the_docs_loads(block: str, tmp_path: Path) -> None:
    files = {"a.md": "", "q/q.json": block}
    if block == ORDER:
        files["q/q.py"] = FILLS
    write_folder(
        tmp_path,
        {
            "a": {
                "kind": "decision",
                "model": "judge",
                "instructions": "a.md",
                "questions": ["q/*.json"],
            }
        },
        files=files,
    )
    spec = load_folder(tmp_path)["a"]
    if block == ORDER:
        assert list(spec.questions) == [] and list(spec.scripted) == ["q"]
        return
    question = spec.questions["q"]
    assert question.instructions and question.type == json.loads(block)["type"]


async def test_the_question_script_of_the_docs_fills_the_customer_s_orders_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORDERS_URL", (tmp_path / "orders").as_uri())
    namespace = await context(tmp_path, Setup(), storage)
    block = next(block for block in PYTHON if first_line(block) == MATCHES)
    await eval(compile(block, "engine.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
    run = namespace["run"]
    assert isinstance(run, Decision) and run.state == {"message": "It is down", "customer": "c-17"}
    assert run.questions["order"].model_dump()["options"] == {
        "A-1001": "Standing desk, shipped",
        "A-1002": "Lamp",
    }
    assert run.answers["order"].model_dump()["choice"] == "A-1001"


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


def test_the_route_entry_names_an_agent_alone(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    tree(root)
    route = load_folder(root)["frontdesk"].route
    assert route is not None and route.to == {"billing": "billing", "technical": "support"}


async def test_the_tool_of_getting_started_runs_in_its_worker_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "agents"
    tree(root)
    write(tmp_path / "orders/acme/orders/A-1001", json.dumps({"status": "shipped"}))
    monkeypatch.setenv("ORDERS_URL", (tmp_path / "orders").as_uri())
    async with Engine(root, providers=providing(None, None, None, None)) as engine:
        surface = EngineTools(engine, agent="support", scope="acme")
        found = await surface.dispatch("lookup_order", {"order_id": "A-1001"})
    assert found == {"status": "shipped"}


async def test_the_bridge_client_of_the_docs_calls_a_tool_from_a_command_as_written(
    tmp_path: Path, short_root: Path
) -> None:
    root = tmp_path / "agents"
    tree(root)
    amend(root, "coder", tools={"allow": ["lookup_order"]})
    work = short_root
    write(work / "solve.py", CLIENT)

    async def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "shipped", "order": args["order_id"]}

    solve = ToolCall(id="s1", name="Shell", args={"command": f"{sys.executable} solve.py"})
    llm = Model(Reply(tool_calls=[solve], stop_reason="tool_use"))
    async with (
        Workspace(Local(work), scope="acme") as workspace,
        Engine(root, providers=providing(llm, None, None, None)) as engine,
    ):
        run = await engine.run(
            "coder",
            "acme",
            "look it up",
            workspace=workspace,
            tools=Tools([Tool("lookup_order", "Fetch an order", body=lookup_order)]),
        )
    assert isinstance(run, Generation)
    assert [(s.tool, s.args, s.by) for s in run.steps[:1]] == [
        ("lookup_order", {"order_id": "A-1001"}, "s1")
    ]
    assert isinstance(run.steps[1].result, str)
    assert json.loads(run.steps[1].result)["stdout"] == "shipped\n"


SECTIONS = {"version", "workers", "models", "tools", "runtime_tools", "agents"}


def _layers(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [entry, *entry.get("modes", {}).values()]


def _completed(written: Mapping[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = (
        dict(written) if set(written) <= SECTIONS else {"agents": dict(written)}
    )
    document.setdefault("version", 1)
    agents = dict(document.get("agents", {}))
    named = [
        layer["model"] for entry in agents.values() for layer in _layers(entry) if "model" in layer
    ]
    models = document.setdefault("models", {name: {"provider": name} for name in named})
    first = next(iter(models))
    for name in models:
        if name not in named:
            agents[f"on-{name}"] = {
                "kind": "generation",
                "model": name,
                "instructions": f"{name}.md",
            }
    reached = [
        name
        for entry in list(agents.values())
        for layer in _layers(entry)
        for orchestration in [layer.get("orchestration", {})]
        for name in [
            *orchestration.get("delegate_to", []),
            *orchestration.get("route", {}).get("to", {}).values(),
        ]
    ]
    for name in reached:
        agents.setdefault(
            name, {"kind": "generation", "model": first, "instructions": f"{name}.md"}
        )
    document["agents"] = agents
    given = runtime(picked_by(agents), document.get("tools", {}))
    if given and "runtime_tools" not in document:
        document["runtime_tools"] = given
    return document


def _touch(root: Path, path: str, text: str) -> None:
    if not (root / path).exists():
        write(root / path, text)


def _under(pattern: str) -> str:
    parts = [part.replace("**", "x").replace("*", "x") for part in pattern.split("/")]
    return "/".join(parts if PurePosixPath(parts[-1]).suffix else [*parts, "x.md"])


def _stubbed(root: Path, document: Mapping[str, Any]) -> None:
    for name, written in document.get("tools", {}).items():
        tool = (
            {"module": written, "schema": written.removesuffix(".py") + ".json"}
            if isinstance(written, str)
            else written
        )
        _touch(
            root,
            tool["module"],
            f"def {tool.get('function', 'run')}(args, call):\n    return {{}}\n",
        )
        if "schema" in tool:
            schema = {"description": f"The {name} tool", "input_schema": {"type": "object"}}
            _touch(root, tool["schema"], json.dumps(schema))
    for entry in document["agents"].values():
        route = entry.get("orchestration", {}).get("route")
        for layer in _layers(entry):
            if "instructions" in layer:
                _touch(root, layer["instructions"], "")
            files = layer.get("files", {})
            for pattern in [
                *files.get("allow", []),
                *files.get("preload", []),
                *files.get("deny", []),
            ]:
                _touch(root, _under(pattern), "")
            skills = layer.get("skills", {})
            public = skills.get("public", [])
            names = [*(public if isinstance(public, list) else []), *skills.get("deny", [])]
            for pattern in [*skills.get("allow", []), *skills.get("preload", [])]:
                folder, _, last = pattern.rpartition("/")
                held = [last]
                if "*" in last:
                    held = [name for name in names if "/" not in name] or ["x"]
                for name in held:
                    _touch(root, f"{folder}/{name}/SKILL.md", "")
            for pattern in layer.get("questions", []):
                asked = route["by"] if route else "q"
                options = {
                    option: option for option in [*(route["to"] if route else ["a"]), "other"]
                }
                path = pattern.replace("*", asked)
                _touch(
                    root,
                    path,
                    json.dumps({"type": "choice", "instructions": "Which?", "options": options}),
                )


@pytest.mark.parametrize("block", YAML, ids=lambda block: block.splitlines()[0])
def test_every_rfnry_yaml_of_the_docs_loads(block: str, tmp_path: Path) -> None:
    written = yaml.safe_load(block)
    document = _completed(written)
    whole = document == written
    assert whole or "version" not in written, "a whole file of the docs needs no completing"
    write(tmp_path / MANIFEST, block if whole else yaml.safe_dump(document, sort_keys=False))
    _stubbed(tmp_path, document)
    agents = load_folder(tmp_path)
    shown = written.get("agents", {}) if set(written) <= SECTIONS else written
    assert set(shown) <= set(agents)


async def test_the_refund_tool_runs_in_a_worker_from_a_shell_and_under_its_tests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agents"
    tools = root / "billing/tools"
    write(root / MANIFEST, DECLARED)
    write(root / "billing/instructions.md", "Refund a duplicate charge.\n")
    for name, text in {
        "refund.json": REFUND_SCHEMA,
        "refund.py": REFUND,
        "_cli.py": CLI,
        "conftest.py": CONFTEST,
        "refund_test.py": REFUND_TEST,
        "payments.py": (
            "from types import SimpleNamespace\n\n\n"
            "async def refund(scope, charge_id, reference=None):\n"
            "    return SimpleNamespace(id='re_1', status='succeeded')\n"
        ),
    }.items():
        write(tools / name, text)
    refunded = {"refund_id": "re_1", "status": "succeeded"}
    async with Engine(root, providers=provided({"anthropic": Model()})) as engine:
        surface = EngineTools(engine, agent="billing", scope="acme")
        assert await surface.dispatch("refund", {"charge_id": "ch_81"}) == refunded

    def shell(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *argv], cwd=root, capture_output=True, text=True, check=False
        )

    program, *argv = shlex.split(
        next(body for lang, body in BLOCKS if lang == "sh" and "--call" in body)
    )
    assert program == "python" and argv[-2] == "--call"
    ran = shell(*argv)
    assert ran.returncode == 0 and json.loads(ran.stdout) == refunded, ran.stderr
    refused = shell(*argv[:-1], "acme")
    assert refused.returncode != 0 and refused.stdout == ""
    tested = shell("-m", "pytest", "-q", "-p", "no:cacheprovider", "billing/tools")
    assert tested.returncode == 0 and "2 passed" in tested.stdout, tested.stdout


def test_the_entrypoint_of_the_docs_is_sh_and_keeps_the_token_out_of_git_s_arguments(
    tmp_path: Path,
) -> None:
    script = next(body for lang, body in BLOCKS if lang == "sh" and body.startswith("#!/bin/sh"))
    write(tmp_path / "entrypoint.sh", script)
    checked = subprocess.run(
        ["sh", "-n", str(tmp_path / "entrypoint.sh")], capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr
    commands = [line for line in script.splitlines() if line.startswith("git ")]
    assert commands and not [line for line in commands if "GIT_AUTH_TOKEN" in line]
    assert script.splitlines()[-2].startswith("unset GIT_AUTH_TOKEN ")
