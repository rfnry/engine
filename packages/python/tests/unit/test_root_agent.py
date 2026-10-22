import ast
from pathlib import Path

import pytest

import rfnry_engine
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Reply, Text, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tree import load_tree

from .fakes import Scripted, text_reply


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def orchestra(root: Path, front: str = "agents: [billing, triage]\n") -> Path:
    write(
        root / "AGENT.md",
        f"---\nname: desk\nkind: generation\nmodel: strong\ndescription: Leads\n{front}---\nYou lead.\n",
    )
    write(root / "playbook.md", "---\ndescription: How we work\n---\nBe kind.\n")
    write(root / "policies/returns.md", "---\ndescription: Returns\n---\n30 days.\n")
    write(
        root / "examples/old/AGENT.md",
        "---\nname: old\nkind: generation\nmodel: strong\n---\nA sample.\n",
    )
    write(
        root / "billing/AGENT.md",
        "---\nname: billing\nkind: generation\nmodel: strong\ndescription: Bills\nreads: [policies]\n---\nYou bill.\n",
    )
    write(root / "billing/rates.md", "---\ndescription: Rates\n---\n$1.\n")
    write(
        root / "triage/AGENT.md",
        "---\nname: triage\nkind: decision\nmodel: judge\ndescription: Sorts\n---\nSort.\n",
    )
    write(root / "triage/questions/urgent.md", "---\ntype: check\n---\nUrgent?\n")
    return root


def test_a_root_agent_has_its_sub_agents_beside_it(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    tree = load_tree(root)
    assert list(tree) == ["desk", "billing", "triage"]
    lead = tree["desk"]
    assert lead.listing.splitlines() == [
        "examples/",
        "  old/",
        "    AGENT.md",
        "playbook.md — How we work",
        "policies/",
        "  returns.md — Returns",
    ]
    assert "../policies/\n  returns.md — Returns" in tree["billing"].listing
    assert tree["triage"].kind == "decision"


def test_a_root_agent_preloads_its_folder_without_its_sub_agents(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk", "agents: [billing]\npreload: true\n")
    sent = "".join(p.text for p in load_tree(root)["desk"].preload if isinstance(p, Text))
    assert "<<<playbook.md>>>" in sent and "<<<policies/returns.md>>>" in sent
    assert "rates.md" not in sent and "You bill." not in sent


async def test_a_root_agent_delegates_to_its_sub_agents(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    llm = Scripted(
        Reply(
            tool_calls=[
                ToolCall(id="d1", name="Delegate", args={"agent": "billing", "input": "x"})
            ],
            stop_reason="tool_use",
        ),
        text_reply("billed"),
        text_reply("done"),
    )
    run = await Engine(root, generation=llm).run("desk", "acme", "Charge them.")
    assert isinstance(run, Generation) and run.reply == "done"
    assert [child.agent for child in run.children] == ["billing"]


def test_a_sub_agent_named_like_the_root_agent_is_refused(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    write(
        root / "desk/AGENT.md", "---\nname: desk\nkind: generation\nmodel: strong\n---\nMe again.\n"
    )
    with pytest.raises(ConfigurationError, match="both name the agent 'desk'; rename one"):
        load_tree(root)


TREE_MODULES = ("tree.py", "frontmatter.py", "folders.py", "tools/declared.py")


def test_every_tree_error_says_its_fix() -> None:
    source = Path(rfnry_engine.__file__).parent
    silent = []
    for module in TREE_MODULES:
        path = source / module
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "ConfigurationError"
            ):
                continue
            said = ast.unparse(node.exc.args[0])
            if "; " not in said and "{exc}" not in said:
                silent.append(f"{module}:{node.lineno}: {said}")
    assert silent == []


async def test_a_linked_agent_folder_is_named_by_its_frontmatter_not_its_link(
    tmp_path: Path,
) -> None:
    write(
        tmp_path / "elsewhere/payments/AGENT.md",
        "---\nname: billing\nkind: generation\nmodel: strong\n---\nYou bill.\n",
    )
    (tmp_path / "root").mkdir()
    (tmp_path / "root/billing").symlink_to(
        tmp_path / "elsewhere/payments", target_is_directory=True
    )
    engine = Engine(tmp_path / "root", generation=Scripted(text_reply("ok")))
    assert list(engine.agents) == ["billing"]
    run = await engine.run("billing", "acme", "x")
    assert run.agent == "billing"


def test_a_linked_sub_agent_keeps_its_own_name_beside_the_root(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    write(
        tmp_path / "elsewhere/desk/AGENT.md",
        "---\nname: kid\nkind: generation\nmodel: strong\n---\nA kid.\n",
    )
    (root / "kid").symlink_to(tmp_path / "elsewhere/desk", target_is_directory=True)
    tree = load_tree(root)
    assert tree["kid"].name == "kid" and tree["desk"].root == root.resolve()


def test_hidden_and_declaration_folders_are_never_sub_agents(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    write(
        root / ".draft/AGENT.md",
        "---\nname: draft\nkind: generation\nmodel: strong\n---\nNot an agent.\n",
    )
    write(root / "tasks/AGENT.md", "Not an agent: a task of the root agent.\n")
    write(root / "tasks/close.md", "Close the ticket.\n")
    tree = load_tree(root)
    assert list(tree) == ["desk", "billing", "triage"]
    assert sorted(tree["desk"].tasks) == ["AGENT", "close"]
    write(
        root / "billing/AGENT.md",
        "---\nname: billing\nkind: generation\nmodel: strong\nreads: [tasks]\n---\nYou bill.\n",
    )
    with pytest.raises(ConfigurationError, match="reads unknown folder 'tasks'"):
        load_tree(root)


def test_a_hidden_folder_is_never_an_agent_of_a_tree(tmp_path: Path) -> None:
    write(
        tmp_path / ".draft/AGENT.md",
        "---\nname: draft\nkind: generation\nmodel: strong\n---\nHidden.\n",
    )
    write(
        tmp_path / "live/AGENT.md", "---\nname: live\nkind: generation\nmodel: strong\n---\nLive.\n"
    )
    assert list(load_tree(tmp_path)) == ["live"]


def test_a_plain_root_is_told_to_pass_tools_on_the_run(tmp_path: Path) -> None:
    write(tmp_path / "notes.md", "Notes.\n")

    async def body(args: dict[str, object]) -> str:
        return "x"

    with pytest.raises(ConfigurationError, match=r"run\(tools=\)"):
        Engine(tmp_path, generation=Scripted(), tools=[rfnry_engine.Tool("x", "X", body=body)])


def test_an_unknown_key_beside_name_is_named_in_the_fix_too(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", "---\nname: a\nkind: generation\nmodel: strong\n---\nHi.\n")
    write(
        tmp_path / "a/tools/fetch.md",
        "---\nname: fetch\ndescription: Fetch\nwrokspace: true\n---\n",
    )
    with pytest.raises(ConfigurationError) as refused:
        load_tree(tmp_path)
    said = str(refused.value)
    assert "remove the name: line" in said and "it takes description" in said


def test_a_decision_task_file_is_told_where_it_goes(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", "---\nname: a\nkind: decision\nmodel: judge\n---\nSort.\n")
    write(tmp_path / "a/questions/q.md", "---\ntype: check\n---\nIs it?\n")
    write(tmp_path / "a/tasks/grade.md", "Grade it.\n")
    with pytest.raises(ConfigurationError, match=r"; move it to tasks/grade/"):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("---\nname: a\nmodel: strong\ndescription: No kind\n---\nHi.\n", "kind: field required"),
        (
            "---\nname: a\nkind: generator\nmodel: strong\n---\nHi.\n",
            "kind: input should be 'generation' or 'decision'",
        ),
    ],
)
def test_an_invalid_field_is_said_plainly(tmp_path: Path, text: str, said: str) -> None:
    write(tmp_path / "a/AGENT.md", text)
    with pytest.raises(ConfigurationError) as refused:
        load_tree(tmp_path)
    assert said in str(refused.value) and "pydantic" not in str(refused.value)
