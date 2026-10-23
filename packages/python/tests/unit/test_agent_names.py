from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.tree import load_tree


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def test_an_agent_is_named_by_its_frontmatter_not_its_folder(tmp_path: Path) -> None:
    write(tmp_path / "support_v2/AGENT.md", agent_md("support"))
    tree = load_tree(tmp_path)
    assert list(tree) == ["support"]
    info = Engine(tmp_path, models={}).agents["support"]
    assert (info.name, info.model, info.path) == ("support", "strong", "support_v2")


@pytest.mark.parametrize(
    ("front", "said"),
    [
        ("kind: generation\nmodel: strong\n", "name: field required"),
        ("name: a\nkind: generation\n", "model: field required"),
        ("name: ''\nkind: generation\nmodel: strong\n", "name:"),
        ("name: ' a'\nkind: generation\nmodel: strong\n", "name:"),
        ("name: .a\nkind: generation\nmodel: strong\n", "name:"),
        ("name: a/b\nkind: generation\nmodel: strong\n", "name:"),
        ("name: a\\b\nkind: generation\nmodel: strong\n", "name:"),
        ("name: a\nkind: generation\nmodel: ' '\n", "model:"),
        ("name: a\nkind: generation\nmodel: ' fast'\n", "model:"),
        ("name: a\nkind: generation\nmodel: 4\n", "model:"),
    ],
)
def test_an_agent_md_without_a_plain_name_and_model_is_refused(
    tmp_path: Path, front: str, said: str
) -> None:
    write(tmp_path / "a/AGENT.md", f"---\n{front}---\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(tmp_path / "a/AGENT.md") in str(caught.value) and said in str(caught.value)


def test_two_agents_that_share_a_name_are_refused_with_both_paths(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("same"))
    write(tmp_path / "b/AGENT.md", agent_md("same"))
    with pytest.raises(
        ConfigurationError, match="both name the agent 'same'; rename one"
    ) as caught:
        load_tree(tmp_path)
    assert str(tmp_path / "a/AGENT.md") in str(caught.value)
    assert str(tmp_path / "b/AGENT.md") in str(caught.value)


def test_the_root_agent_and_one_in_its_folder_that_share_a_name_are_refused_with_both_paths(
    tmp_path: Path,
) -> None:
    write(tmp_path / "AGENT.md", agent_md("desk"))
    write(tmp_path / "billing/AGENT.md", agent_md("desk"))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(tmp_path / "AGENT.md") in str(caught.value)
    assert str(tmp_path / "billing/AGENT.md") in str(caught.value)


def test_the_engine_takes_no_name(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    with pytest.raises(TypeError):
        Engine(tmp_path, name="a", models={})  # type: ignore[call-arg]


def test_the_model_line_is_part_of_the_agent_s_fingerprint(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", model="strong"))
    before = load_tree(tmp_path)["a"].fingerprint
    write(tmp_path / "a/AGENT.md", agent_md("a", model="fast"))
    assert load_tree(tmp_path)["a"].fingerprint != before
