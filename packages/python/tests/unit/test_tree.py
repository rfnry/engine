from pathlib import Path

import pytest

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Choice, Score
from rfnry_engine.tree import load_agent, load_tree, validate_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


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
    (tmp_path / "a" / "tools" / "x.md").write_text("---\nname: x\ninput: {q: {type: string}}\n---\n")
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
    (tmp_path / "d" / "questions" / "q.md").write_text("---\ntype: choice\noptions: {x: X}\n---\n?\n")
    with pytest.raises(ConfigurationError):
        load_agent(tmp_path / "d")


def _agent(root: Path, name: str, front: str, questions: dict[str, str] | None = None, tools: tuple[str, ...] = ()) -> None:
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
        ("kind: decision\nworkspace: true\n", {"q": CHECK}, (), "decision agent 'a' cannot declare workspace: true"),
        ("kind: generation\n", {"q": CHECK}, (), "generation agent 'a' cannot declare questions/"),
        ("kind: generation\nroute: {by: q, to: {x: a}}\n", {}, (), "generation agent 'a' cannot declare route:"),
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
        ("kind: decision\nroute: {by: urgent, to: {x: b}}\n", "route of 'a' must name a choice question"),
        ("kind: decision\nroute: {by: q, to: {z: b}}\n", "route of 'a' maps unknown option 'z'"),
        ("kind: decision\nagents: [ghost]\n", "agent 'a' names unknown agent 'ghost'"),
    ],
)
def test_routes_and_agents_must_resolve(tmp_path: Path, front: str, message: str) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE, "urgent": CHECK})
    _agent(tmp_path, "b", "kind: generation\n")
    with pytest.raises(ConfigurationError, match=f"^tree: {message}$"):
        validate_tree(load_tree(tmp_path))
