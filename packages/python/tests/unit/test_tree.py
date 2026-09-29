from pathlib import Path

import pytest

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Check, Choice, Score
from rfnry_engine.tree import load_agent, load_tree, validate_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"
EXAMPLES = Path(__file__).parents[4] / "examples"


def test_load_tree_finds_every_agent_with_its_kind() -> None:
    agents = load_tree(FIXTURES)
    assert {n: a.kind for n, a in agents.items()} == {
        "support": "generation",
        "billing": "generation",
        "triage": "decision",
        "frontdesk": "decision",
        "judge": "decision",
        "teacher": "generation",
    }


def test_generation_agent_carries_body_agents_tools_tasks() -> None:
    support = load_tree(FIXTURES)["support"]
    assert support.body.startswith("You are the support desk")
    assert support.agents == ["triage"]
    assert support.tasks == ["triage"]
    contract = support.tools["lookup_order"]
    assert contract.input is not None and contract.input["order_id"].required is True
    assert contract.description is not None and "Returns the order as JSON." in contract.description
    adoption = support.tools["now"]
    assert adoption.description is None and adoption.input is None


def test_a_tool_file_is_a_contract_or_a_name(tmp_path: Path) -> None:
    (tmp_path / "a" / "tools").mkdir(parents=True)
    (tmp_path / "a" / "AGENT.md").write_text("---\nname: a\nkind: generation\n---\n")
    (tmp_path / "a" / "tools" / "x.md").write_text(
        "---\nname: x\ninput: {q: {type: string}}\n---\n"
    )
    with pytest.raises(ConfigurationError, match="description"):
        load_agent(tmp_path / "a")


def test_decision_agent_carries_typed_questions_and_route() -> None:
    agents = load_tree(FIXTURES)
    severity = agents["triage"].questions["severity"]
    assert isinstance(severity, Score) and len(severity.levels) == 4
    intent = agents["frontdesk"].questions["intent"]
    assert isinstance(intent, Choice) and set(intent.options) == {"billing", "technical", "other"}
    route = agents["frontdesk"].route
    assert route is not None and route.by == "intent" and route.min_confidence == 0.6


def test_validate_tree_accepts_the_fixture() -> None:
    validate_tree(load_tree(FIXTURES))


def test_single_agent_root(tmp_path: Path) -> None:
    (tmp_path / "solo").mkdir()
    (tmp_path / "solo" / "AGENT.md").write_text("---\nname: solo\nkind: generation\n---\nHi\n")
    assert list(load_tree(tmp_path / "solo")) == ["solo"]


def test_name_must_match_directory(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text("---\nname: b\nkind: generation\n---\n")
    with pytest.raises(ConfigurationError):
        load_agent(tmp_path / "a")


def test_route_must_point_at_existing_agents(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "questions").mkdir()
    (tmp_path / "d" / "AGENT.md").write_text(
        "---\nname: d\nkind: decision\nroute: {by: q, to: {x: ghost}}\n---\n"
    )
    (tmp_path / "d" / "questions" / "q.md").write_text(
        "---\ntype: choice\noptions: {x: X, y: Y}\n---\nWhich?\n"
    )
    with pytest.raises(ConfigurationError, match="ghost"):
        validate_tree(load_tree(tmp_path))


def test_question_with_one_option_is_refused(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "questions").mkdir()
    (tmp_path / "d" / "AGENT.md").write_text("---\nname: d\nkind: decision\n---\n")
    (tmp_path / "d" / "questions" / "q.md").write_text(
        "---\ntype: choice\noptions: {x: X}\n---\n?\n"
    )
    with pytest.raises(ConfigurationError):
        load_agent(tmp_path / "d")


def _agent(
    root: Path,
    name: str,
    front: str,
    questions: dict[str, str] | None = None,
    tools: tuple[str, ...] = (),
) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "AGENT.md").write_text(f"---\nname: {name}\n{front}---\n")
    for qid, text in (questions or {}).items():
        (root / name / "questions").mkdir(exist_ok=True)
        (root / name / "questions" / f"{qid}.md").write_text(text)
    for tool in tools:
        (root / name / "tools").mkdir(exist_ok=True)
        (root / name / "tools" / f"{tool}.md").write_text(f"---\nname: {tool}\n---\n")


CHOICE = "---\ntype: choice\noptions: {x: X, y: Y}\n---\nWhich?\n"
CHECK = "---\ntype: check\n---\nIs it?\n"


@pytest.mark.parametrize(
    ("front", "questions", "tools", "message"),
    [
        ("kind: decision\n", {}, (), "decision agent 'a' has no questions/"),
        ("kind: decision\n", {"q": CHECK}, ("t",), "decision agent 'a' cannot declare tools/"),
        (
            "kind: decision\nworkspace: true\n",
            {"q": CHECK},
            (),
            "decision agent 'a' cannot declare workspace: true",
        ),
        ("kind: generation\n", {"q": CHECK}, (), "generation agent 'a' cannot declare questions/"),
        (
            "kind: generation\nroute: {by: q, to: {x: a}}\n",
            {},
            (),
            "generation agent 'a' cannot declare route:",
        ),
    ],
)
def test_each_kind_holds_only_what_it_can_use(
    tmp_path: Path, front: str, questions: dict[str, str], tools: tuple[str, ...], message: str
) -> None:
    _agent(tmp_path, "a", front, questions, tools)
    with pytest.raises(ConfigurationError, match=f"^tree: {message}$"):
        load_agent(tmp_path / "a")


@pytest.mark.parametrize(
    ("front", "message"),
    [
        (
            "kind: decision\nroute: {by: urgent, to: {x: b}}\n",
            "route of 'a' must name a choice question",
        ),
        ("kind: decision\nroute: {by: q, to: {z: b}}\n", "route of 'a' maps unknown option 'z'"),
        ("kind: decision\nagents: [ghost]\n", "agent 'a' names unknown agent 'ghost'"),
    ],
)
def test_routes_and_agents_must_resolve(tmp_path: Path, front: str, message: str) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE, "urgent": CHECK})
    _agent(tmp_path, "b", "kind: generation\n")
    with pytest.raises(ConfigurationError, match=f"^tree: {message}$"):
        validate_tree(load_tree(tmp_path))


@pytest.mark.parametrize(
    ("front", "tool", "key", "where"),
    [
        ("kind: generation\nagent: [b]\n", None, "agent", "AGENT.md"),
        ("kind: generation\nworkpsace: true\n", None, "workpsace", "AGENT.md"),
        (
            "kind: decision\nroute: {by: q, to: {x: b}, min_confidance: 0.9}\n",
            None,
            "route.min_confidance",
            "AGENT.md",
        ),
        (
            "kind: generation\n",
            "---\nname: t\ndescription: T\nsummary: S\n---\n",
            "summary",
            "t.md",
        ),
        (
            "kind: generation\n",
            "---\nname: t\ndescription: T\ninput:\n  q: {type: string, requried: true}\n---\n",
            "input.q.requried",
            "t.md",
        ),
    ],
)
def test_an_unknown_frontmatter_key_is_refused_by_name(
    tmp_path: Path, front: str, tool: str | None, key: str, where: str
) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE} if "decision" in front else {})
    if tool is not None:
        (tmp_path / "a" / "tools").mkdir()
        (tmp_path / "a" / "tools" / "t.md").write_text(tool)
    path = tmp_path / "a" / ("tools/t.md" if tool is not None else "AGENT.md")
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "a")
    assert str(refused.value) == f"frontmatter: unknown key {key!r} in {path}"
    assert str(refused.value).endswith(where)


def test_only_true_and_false_are_booleans_in_frontmatter(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\nworkspace: true\n")
    (tmp_path / "a" / "tools").mkdir()
    (tmp_path / "a" / "tools" / "t.md").write_text(
        "---\nname: t\ndescription: T\ninput:\n"
        "  q: {type: string, required: true}\n"
        "  r: {type: string, required: yes}\n"
        "  s: {type: string, required: False}\n"
        "---\n"
    )
    spec = load_agent(tmp_path / "a")
    assert spec.workspace is True and spec.tools["t"].input is not None
    assert {k: f.required for k, f in spec.tools["t"].input.items()} == {
        "q": True,
        "r": True,
        "s": False,
    }
    choice = "---\ntype: choice\noptions: {yes: Agrees, no: Disagrees, off: Unrelated}\n---\n?\n"
    _agent(tmp_path, "d", "kind: decision\n", {"q": choice})
    question = load_agent(tmp_path / "d").questions["q"]
    assert isinstance(question, Choice)
    assert question.options == {"yes": "Agrees", "no": "Disagrees", "off": "Unrelated"}


def test_every_example_tree_loads_with_its_checks_criteria() -> None:
    roots = sorted(EXAMPLES.glob("*/agents"))
    assert roots
    for root in roots:
        agents = load_tree(root)
        validate_tree(agents)
        for spec in agents.values():
            for qid, question in spec.questions.items():
                if not isinstance(question, Check):
                    continue
                text = (spec.root / "questions" / f"{qid}.md").read_text(encoding="utf-8")
                assert (question.yes is not None) == ("\nyes:" in text), (root, qid)
                assert (question.no is not None) == ("\nno:" in text), (root, qid)


def test_an_unknown_key_in_a_question_is_refused_by_name(tmp_path: Path) -> None:
    typo = "---\ntype: choice\noptoins: {x: X, y: Y}\n---\nWhich?\n"
    _agent(tmp_path, "d", "kind: decision\n", {"q": typo})
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "d")
    path = tmp_path / "d" / "questions" / "q.md"
    assert str(refused.value) == f"frontmatter: unknown key 'optoins' in {path}"


def test_a_question_takes_structure_through_its_frontmatter(tmp_path: Path) -> None:
    choice = (
        "---\ntype: choice\n"
        "instructions:\n  question: Which team handles `ticket`?\n  ticket: {id: 7, vip: true}\n"
        "options:\n"
        "  billing:\n    what: Charges\n    examples: [charged twice, yes, on]\n"
        "  orders: [Delivery, Returns]\n"
        "  other:\n"
        "  legal: Contracts\n"
        "---\n"
    )
    score = (
        "---\ntype: score\nlevels:\n  - {summary: One change, signals: [a single fix]}\n"
        "  - [Several]\n  - Many\n---\nHow focused is the change?\n"
    )
    check = (
        "---\ntype: check\ninstructions:\n  - Does `message` ask for a credential?\n  - message\n"
        "yes: {what: Asks for a password, examples: [Reply with your PIN]}\n"
        "no: [No credential is requested]\n---\n"
    )
    _agent(tmp_path, "d", "kind: decision\n", {"team": choice, "scope": score, "creds": check})
    assert load_agent(tmp_path / "d").questions == {
        "creds": Check(
            instructions=["Does `message` ask for a credential?", "message"],
            yes={"what": "Asks for a password", "examples": ["Reply with your PIN"]},
            no=["No credential is requested"],
        ),
        "scope": Score(
            instructions="How focused is the change?",
            levels=[{"summary": "One change", "signals": ["a single fix"]}, ["Several"], "Many"],
        ),
        "team": Choice(
            instructions={
                "question": "Which team handles `ticket`?",
                "ticket": {"id": 7, "vip": True},
            },
            options={
                "billing": {"what": "Charges", "examples": ["charged twice", "yes", "on"]},
                "orders": ["Delivery", "Returns"],
                "other": None,
                "legal": "Contracts",
            },
        ),
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "---\ntype: check\ninstructions:\n  question: Is it?\n---\nIs it?\n",
            "gives instructions both in its frontmatter and in its body",
        ),
        ("---\ntype: check\n---\n\n", "gives no instructions, in its body or its frontmatter"),
        (
            "---\ntype: check\ninstructions: {}\n---\n",
            "gives no instructions, in its body or its frontmatter",
        ),
    ],
)
def test_a_question_has_its_instructions_in_its_frontmatter_or_its_body(
    tmp_path: Path, text: str, message: str
) -> None:
    _agent(tmp_path, "d", "kind: decision\n", {"q": text})
    path = tmp_path / "d" / "questions" / "q.md"
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "d")
    assert str(refused.value) == f"tree: question {path} {message}"


@pytest.mark.parametrize(
    "text",
    [
        "---\ntype: check\ninstructions: Is it?\n---\n",
        "---\ntype: score\nlevels: [low, ~]\n---\nHow?\n",
        "---\ntype: score\nlevels: [low, 2]\n---\nHow?\n",
        "---\ntype: choice\noptions: {a: A, b: 2026-01-01}\n---\nWhich?\n",
    ],
)
def test_structure_in_frontmatter_is_an_object_or_a_list_of_json(tmp_path: Path, text: str) -> None:
    _agent(tmp_path, "d", "kind: decision\n", {"q": text})
    with pytest.raises(ConfigurationError, match="^frontmatter: invalid question "):
        load_agent(tmp_path / "d")
