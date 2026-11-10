import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Check, Choice
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Answering, Scripted, delegating, text_reply
from .trees import write_folder

GENERATION: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "a.md"}
DECISION: dict[str, Any] = {"kind": "decision", "model": "j", "instructions": "a.md"}
URGENT = json.dumps({"type": "check", "instructions": "Is it urgent?"})
AREA = json.dumps(
    {"type": "choice", "instructions": "Which area?", "options": {"other": None, "billing": None}}
)
FILES = {"a.md": "Work.\n", "q/area.json": AREA}
Reaching = Callable[[str], dict[str, Any]]


def _delegating(target: str) -> dict[str, Any]:
    return {**GENERATION, "orchestration": {"delegate_to": [target]}}


def _routing(target: str) -> dict[str, Any]:
    route = {"by": "area", "to": {"billing": target}}
    return {**DECISION, "questions": ["q/*.json"], "orchestration": {"route": route}}


def _on_a_mode(agent: dict[str, Any]) -> dict[str, Any]:
    entry = dict(agent)
    said = entry.pop("orchestration")
    return {**entry, "modes": {"x": {"orchestration": said}}}


def _refused(root: Path, pattern: str) -> str:
    with pytest.raises(ConfigurationError) as raised:
        load_folder(root)
    said = str(raised.value)
    assert re.search(pattern, said), said
    return said


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
async def test_an_agent_runs_on_its_instructions_file(tmp_path: Path, newline: str) -> None:
    text = "---\ndescription: Says a\nvendor: kept out\n---\nYou are a.\n".replace("\n", newline)
    folder = write_folder(tmp_path, {"a": GENERATION}, files={"a.md": text})
    model = Scripted(text_reply("done"))
    run = await Engine(folder, models={"m": model}).run("a", "acme", "hi")
    assert isinstance(run, Generation) and run.reply == "done"
    [(system, _, _)] = model.calls
    assert "You are a." in system
    assert "vendor" not in system and "---" not in system


@pytest.mark.parametrize("path", [".claude/agents/reviewer.md", "AGENTS.md", "deep/down/here.md"])
async def test_any_markdown_path_is_instructions(tmp_path: Path, path: str) -> None:
    folder = write_folder(
        tmp_path, {"a": {**GENERATION, "instructions": path}}, files={path: "Review it.\n"}
    )
    model = Scripted(text_reply("done"))
    await Engine(folder, models={"m": model}).run("a", "acme", "hi")
    assert "Review it." in model.calls[0][0]


def test_the_description_falls_back_to_the_frontmatter(tmp_path: Path) -> None:
    fronted = "---\ndescription: From the frontmatter\n---\nBody\n"
    folder = write_folder(
        tmp_path,
        {
            "written": {**GENERATION, "instructions": "front.md", "description": "From the file"},
            "falls": {
                **GENERATION,
                "instructions": "front.md",
                "modes": {"own": {"instructions": "mode.md"}},
            },
            "plain": {**GENERATION, "instructions": "plain.md"},
        },
        files={
            "front.md": fronted,
            "mode.md": fronted.replace("the frontmatter", "the mode's frontmatter"),
            "plain.md": "Body\n",
        },
    )
    agents = load_folder(folder)
    assert agents["written"].description == "From the file"
    assert agents["falls"].description == "From the frontmatter"
    assert agents["falls"].modes["own"].description == "From the mode's frontmatter"
    assert agents["plain"].description == ""


def test_frontmatter_is_read_only_for_a_description_the_file_lacks(tmp_path: Path) -> None:
    broken = "---\ndescription: [never closed\n---\nBody\n"
    folder = write_folder(
        tmp_path, {"a": {**GENERATION, "description": "Written"}}, files={"a.md": broken}
    )
    assert load_folder(folder)["a"].body == "Body"
    write_folder(folder, {"a": GENERATION}, files={"a.md": broken})
    _refused(folder, r"^frontmatter: invalid YAML in .*a\.md")


async def test_a_decision_agent_answers_its_questions(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "instructions": "triage/t.md", "questions": ["triage/*.json"]}},
        files={"triage/t.md": "Sort it.\n", "triage/urgent.json": URGENT, "triage/area.json": AREA},
    )
    model = Answering()
    run = await Engine(folder, models={"j": model}).run("t", "acme", "hello")
    assert isinstance(run, Decision)
    assert set(run.answers) == {"urgent", "area"}
    assert run.questions["urgent"] == Check(instructions="Is it urgent?")
    (folder / "triage/orphan.py").write_text(
        "def run(input, call):\n    return {}\n", encoding="utf-8"
    )
    _refused(folder, r"^tree: .*orphan\.py has no orphan\.json beside it")


async def test_a_question_script_fills_its_question_in_a_worker(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "questions": ["q/*.json"]}},
        files={
            "a.md": "Sort.\n",
            "q/order.json": '{"type": "choice", "instructions": "Which order?"}',
            "q/_names.py": "FIRST = 'first'\n",
            "q/order.py": "import _names\n\n\ndef run(input, call):\n    return {'options': {_names.FIRST: 'First', call['scope']: 'Scope'}}\n",
        },
    )
    async with Engine(folder, models={"j": Answering()}) as engine:
        run = await engine.run("t", "acme", "which?")
    assert isinstance(run, Decision)
    question = run.questions["order"]
    assert isinstance(question, Choice) and list(question.options) == ["first", "acme"]


def test_only_the_files_a_pattern_reaches_are_asked(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "questions": ["q/urgent.json"]}},
        files={
            "a.md": "Sort.\n",
            "q/urgent.json": URGENT,
            "q/urgent.md": "Notes on this question.\n",
            "q/filled.json": '{"type": "choice", "instructions": "Which?"}',
            "q/filled.py": "this is not python\n",
            "q/broken.json": "{",
            "q/notes.md": "Not a question.\n",
        },
    )
    spec = load_folder(folder)["t"]
    assert list(spec.questions) == ["urgent"] and spec.scripted == {}


def test_a_pattern_skips_hidden_and_helper_files_unless_it_names_them(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "t": {
                **DECISION,
                "questions": ["q/*.json", "q/*/*.json", ".held/q/.named.json", ".held/**/*.json"],
            }
        },
        files={
            "a.md": "Sort.\n",
            "q/urgent.json": URGENT,
            "q/.hidden.json": "{",
            "q/_helper.json": "{",
            "q/.sub/inner.json": "{",
            "q/sub/area.json": AREA,
            ".held/q/.named.json": URGENT.replace("urgent", "named"),
            ".held/deep/pressing.json": URGENT,
            ".held/.held/secret.json": "{",
        },
    )
    assert sorted(load_folder(folder)["t"].questions) == [".named", "area", "pressing", "urgent"]


@pytest.mark.parametrize("written", ["q/*.txt", "gone/*.json", "q", "q/gone.json"])
def test_a_pattern_reaching_no_question_is_refused(tmp_path: Path, written: str) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "questions": [written]}},
        files={"a.md": "Sort.\n", "q/urgent.json": URGENT, "q/notes.txt": "x\n"},
    )
    said = _refused(folder, r"^compiled: agent 't' .* reach no question file")
    assert repr(written) in said


def test_a_pattern_reaching_markdown_is_refused_as_a_question(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "questions": ["q/*"]}},
        files={"a.md": "Sort.\n", "q/urgent.md": "Is it urgent?\n"},
    )
    _refused(folder, r"^tree: question .*urgent\.md is markdown")


def test_one_file_reached_twice_is_one_question_and_two_files_of_one_name_are_refused(
    tmp_path: Path,
) -> None:
    files = {"a.md": "Sort.\n", "q/urgent.json": URGENT, "r/urgent.json": URGENT}
    once = {"t": {**DECISION, "questions": ["q/*.json", "q/urgent.json"]}}
    assert list(load_folder(write_folder(tmp_path, once, files=files))["t"].questions) == ["urgent"]
    twice = {"t": {**DECISION, "questions": ["q/*.json", "r/*.json"]}}
    said = _refused(
        write_folder(tmp_path, twice, files=files), r"^compiled: agent 't' asks 'urgent' twice"
    )
    assert "q/urgent.json" in said and "r/urgent.json" in said


def test_a_mode_adds_its_questions_to_the_entrys(tmp_path: Path) -> None:
    agent = {
        **DECISION,
        "questions": ["q/*.json"],
        "modes": {"x": {"questions": ["r/*.json"]}},
    }
    folder = write_folder(tmp_path, {"t": agent}, files={**FILES, "r/urgent.json": URGENT})
    spec = load_folder(folder)["t"]
    assert list(spec.questions) == ["area"]
    assert sorted(spec.modes["x"].questions) == ["area", "urgent"]


@pytest.mark.parametrize("what", ["instructions", "questions"])
@pytest.mark.parametrize("through", ["file", "folder"])
def test_nothing_read_at_load_leads_out_of_the_root(
    tmp_path: Path, what: str, through: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("Secret.\n", encoding="utf-8")
    (outside / "secret.json").write_text(URGENT, encoding="utf-8")
    root = tmp_path / "agents"
    root.mkdir()
    (root / "link").mkdir()
    name = "secret.md" if what == "instructions" else "secret.json"
    if through == "file":
        (root / "link" / name).symlink_to(outside / name)
    else:
        (root / "link").rmdir()
        (root / "link").symlink_to(outside)
    agent = (
        {**DECISION, "instructions": "link/secret.md"}
        if what == "instructions"
        else {**DECISION, "questions": ["link/*.json"]}
    )
    folder = write_folder(root, {"t": agent}, files={"a.md": "Sort.\n"})
    said = _refused(folder, r"^compiled: agent 't' reaches 'link/secret\.[a-z]+'")
    assert "outside the folder root" in said


def test_a_link_inside_the_root_is_followed(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "instructions": "b.md", "questions": ["q/*.json"]}},
        files={"real.md": "Real.\n", "real/urgent.json": URGENT},
    )
    (folder / "b.md").symlink_to(folder / "real.md")
    (folder / "q").mkdir()
    (folder / "q/urgent.json").symlink_to(folder / "real/urgent.json")
    spec = load_folder(folder)["t"]
    assert spec.body == "Real." and list(spec.questions) == ["urgent"]


def test_a_question_script_cannot_import_the_servers_code_from_under_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    root = tmp_path / "agents"
    root.mkdir()
    folder = write_folder(
        root,
        {"t": {**DECISION, "questions": ["q/*.json"]}},
        files={
            "a.md": "Sort.\n",
            "lib/shared_outside.py": "VALUE = 1\n",
            "q/order.json": '{"type": "choice", "instructions": "Which order?"}',
            "q/order.py": "import shared_outside\n\n\ndef run(input, call):\n    return {'options': {'a': 'A'}}\n",
        },
    )
    monkeypatch.syspath_prepend(root / "lib")
    _refused(
        folder,
        r"^tools/folder: .*order\.py imports 'shared_outside' from .*shared_outside\.py, the server's own code",
    )


@pytest.mark.parametrize("through", ["its name", "a link to it", "the file it links to"])
def test_rfnry_yaml_is_never_a_models_to_read(tmp_path: Path, through: str) -> None:
    written = {
        "its name": "rfnry.yaml",
        "a link to it": "b.md",
        "the file it links to": "real.yaml",
    }
    folder = write_folder(tmp_path, {"a": {**GENERATION, "instructions": written[through]}})
    if through == "a link to it":
        (folder / "b.md").symlink_to(folder / "rfnry.yaml")
    if through == "the file it links to":
        (folder / "rfnry.yaml").rename(folder / "real.yaml")
        (folder / "rfnry.yaml").symlink_to(folder / "real.yaml")
    said = _refused(folder, r"^compiled: agent 'a' reaches '[^']+', which is rfnry\.yaml")
    assert repr(written[through]) in said


def test_a_file_name_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "questions": ["q/*.json"]}},
        files={"a.md": "Sort.\n", "q/urgent.json": URGENT},
    )
    named = os.fsencode(folder / "q") + b"/caf\xe9.json"
    with open(named, "w", encoding="utf-8") as file:
        file.write(URGENT)
    _refused(folder, r"^tree: the name of .* is not UTF-8; rename it")


async def test_delegate_to_and_route_name_agents_of_the_file(tmp_path: Path) -> None:
    reachings: list[Reaching] = [_delegating, _routing]
    for reaching in reachings:
        for agent in (reaching("ghost"), _on_a_mode(reaching("ghost"))):
            write_folder(tmp_path, {"a": agent, "b": GENERATION}, files=FILES)
            said = _refused(tmp_path, r"^compiled: (mode 'x' of )?agent 'a' names 'ghost'")
            assert "a, b" in said
    write_folder(tmp_path, {"a": _delegating("b"), "b": GENERATION, "c": GENERATION}, files=FILES)
    model = Scripted(delegating("b", "work"), text_reply("from b"), text_reply("done"))
    run = await Engine(tmp_path, models={"m": model}).run("a", "acme", "go")
    assert isinstance(run, Generation) and run.reply == "done"
    [delegate] = model.calls[0][2]
    assert delegate.name == "Delegate"
    assert delegate.input_schema["properties"]["agent"]["enum"] == ["b"]
    assert [child.reply for child in run.children if isinstance(child, Generation)] == ["from b"]


async def test_a_route_hands_the_input_to_an_agent_of_the_file(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"t": _routing("b"), "b": GENERATION}, files=FILES)
    generating = Scripted(text_reply("answered"))
    run = await Engine(folder, models={"j": Answering(), "m": generating}).run("t", "acme", "x")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.reply == "answered"


def test_a_route_on_an_entry_with_only_modes_is_checked_on_its_modes(tmp_path: Path) -> None:
    def triage(*modes: str) -> dict[str, Any]:
        return {
            "kind": "decision",
            "model": "j",
            "orchestration": {"route": {"by": "area", "to": {"billing": "b"}}},
            "modes": {
                mode: {"instructions": "a.md", "questions": [f"{mode}/*.json"]} for mode in modes
            },
        }

    files = {"a.md": "Sort.\n", "x/area.json": AREA, "y/area.json": AREA, "z/urgent.json": URGENT}
    folder = write_folder(tmp_path, {"t": triage("x", "y"), "b": GENERATION}, files=files)
    spec = load_folder(folder)["t"]
    assert spec.needs_mode and spec.questions == {} and spec.body == ""
    assert list(spec.modes["x"].questions) == ["area"]
    write_folder(folder, {"t": triage("x", "z"), "b": GENERATION}, files=files)
    _refused(folder, r"^compiled: mode 'z' of agent 't' routes by 'area', which is not a choice")


@pytest.mark.parametrize("on_mode", [False, True], ids=["entry", "mode"])
@pytest.mark.parametrize(
    ("route", "said"),
    [
        (
            {"by": "order", "to": {"a": "b"}},
            r"routes by 'order', whose options come from q/order\.py; route by a question whose options are fixed",
        ),
        (
            {"by": "urgent", "to": {"a": "b"}},
            r"routes by 'urgent', which is not a choice question in its questions:",
        ),
        (
            {"by": "gone", "to": {"a": "b"}},
            r"routes by 'gone', which is not a choice question in its questions:",
        ),
        (
            {"by": "area", "to": {"nope": "b"}},
            r"routes the option 'nope', which 'area' does not have; its options are other, billing",
        ),
        (
            {"by": "area", "to": {"billing": "empty"}},
            r"routes to decision agent 'empty', which has no questions: to ask",
        ),
    ],
    ids=["scripted", "not a choice", "no such question", "unknown option", "nothing to ask"],
)
def test_a_refused_route_names_its_layer_and_the_keys_of_the_file(
    tmp_path: Path, route: dict[str, Any], said: str, on_mode: bool
) -> None:
    agent = {**DECISION, "questions": ["q/*.json"], "orchestration": {"route": route}}
    folder = write_folder(
        tmp_path,
        {"t": _on_a_mode(agent) if on_mode else agent, "b": GENERATION, "empty": DECISION},
        files={
            **FILES,
            "q/urgent.json": URGENT,
            "q/order.json": '{"type": "choice", "instructions": "Which order?"}',
            "q/order.py": "def run(input, call):\n    return {'options': {'a': 'A'}}\n",
        },
    )
    who = "mode 'x' of agent 't'" if on_mode else "agent 't'"
    message = _refused(folder, f"^compiled: {who} {said}")
    assert "questions/" not in message


def test_a_route_every_mode_replaces_still_names_agents_of_the_file(tmp_path: Path) -> None:
    replaced = {"by": "area", "to": {"billing": "b"}}
    agent = {
        "kind": "decision",
        "model": "j",
        "questions": ["q/*.json"],
        "orchestration": {"route": {"by": "area", "to": {"billing": "ghost"}}},
        "modes": {"x": {"instructions": "a.md", "orchestration": {"route": replaced}}},
    }
    folder = write_folder(tmp_path, {"t": agent, "b": GENERATION}, files=FILES)
    _refused(folder, r"^compiled: agent 't' names 'ghost'")


@pytest.mark.parametrize("on_mode", [False, True], ids=["entry", "mode"])
@pytest.mark.parametrize("reaching", [_delegating, _routing], ids=["delegate_to", "route"])
@pytest.mark.parametrize("kind", ["generation", "decision"])
def test_an_agent_with_only_modes_cannot_be_a_delegate_target(
    tmp_path: Path, reaching: Reaching, on_mode: bool, kind: str
) -> None:
    only_modes = {
        "kind": kind,
        "model": "m",
        "modes": {"x": {"instructions": "a.md"}},
    }
    agent = _on_a_mode(reaching("picky")) if on_mode else reaching("picky")
    folder = write_folder(tmp_path, {"a": agent, "picky": only_modes}, files=FILES)
    said = _refused(
        folder, r"^compiled: (mode 'x' of )?agent 'a' reaches 'picky', which needs a mode"
    )
    assert "takes none" in said


def test_an_agent_with_instructions_and_modes_can_be_a_delegate_target(tmp_path: Path) -> None:
    target = {**GENERATION, "modes": {"x": {"description": "Mode"}}}
    folder = write_folder(tmp_path, {"a": _delegating("b"), "b": target}, files=FILES)
    assert load_folder(folder)["a"].agents == ["b"]


def test_a_self_reference_loads_as_the_old_loader_lets_it(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": _delegating("a")}, files=FILES)
    assert load_folder(folder)["a"].agents == ["a"]


async def test_workspace_true_needs_a_workspace_on_the_run(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    root.mkdir()
    write_folder(root, {"a": {**GENERATION, "workspace": True}}, files=FILES)
    model = Scripted(text_reply("done"))
    engine = Engine(root, models={"m": model})
    with pytest.raises(ConfigurationError, match="declares workspace: true; pass workspace="):
        await engine.run("a", "acme", "go")
    room = tmp_path / "room"
    room.mkdir()
    await engine.run("a", "acme", "go", workspace=Workspace(Local(room), scope="acme"))
    assert {"Shell", "Write"} <= {tool.name for tool in model.calls[0][2]}


async def test_a_missing_section_gives_nothing(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, files=FILES)
    spec = load_folder(folder)["a"]
    assert spec.keep == [] and spec.folder_tools == {} and spec.skills == {}
    assert spec.listing == "" and spec.preload == [] and spec.withheld is None
    model = Scripted(text_reply("done"))
    given = Tool("given", "A tool of the run", body=None)
    await Engine(folder, models={"m": model}).run("a", "acme", "go", tools=Tools([given]))
    assert model.calls[0][2] == []


def test_each_entry_and_mode_compiles_to_a_full_agent(tmp_path: Path) -> None:
    entry = {
        **GENERATION,
        "description": "Entry",
        "max_model_calls": 9,
        "worker": "w",
        "tools": {"timeout": 3.0, "max_parallel": 2},
        "orchestration": {"delegate_to": ["b"], "max_depth": 1, "max_parallel": 4},
        "modes": {
            "x": {
                "instructions": "x.md",
                "model": "n",
                "workspace": True,
                "max_model_calls": 5,
                "orchestration": {"delegate_to": ["c"]},
            }
        },
    }
    folder = write_folder(
        tmp_path,
        {"a": entry, "b": GENERATION, "c": GENERATION},
        workers={"w": {"processes": 1}},
        files={"a.md": "Entry body.\n", "x.md": "Mode body.\n"},
    )
    agents = load_folder(folder)
    spec = agents["a"]
    assert list(agents) == ["a", "b", "c"]
    assert (spec.root, spec.path, spec.reads, spec.skill_folders) == (folder.resolve(), ".", {}, {})
    assert (spec.name, spec.kind, spec.model, spec.body) == ("a", "generation", "m", "Entry body.")
    assert (spec.needs_mode, spec.mode, spec.worker, spec.workspace) == (False, None, "w", False)
    assert (spec.max_model_calls, spec.tool_timeout, spec.tool_parallel) == (9, 3.0, 2)
    assert (spec.max_depth, spec.delegate_parallel, spec.agents) == (1, 4, ["b"])
    mode = spec.modes["x"]
    assert (mode.name, mode.mode, mode.model, mode.body) == ("a", "x", "n", "Mode body.")
    assert (mode.workspace, mode.max_model_calls, mode.worker) == (True, 5, "w")
    assert (mode.agents, mode.tool_timeout, mode.max_depth) == (["b", "c"], 3.0, 1)
    assert (mode.root, mode.path, mode.modes, mode.needs_mode) == (folder.resolve(), ".", {}, False)
    assert agents["b"].worker == "default" and agents["b"].max_model_calls is None


def test_an_entry_without_instructions_needs_a_mode(tmp_path: Path) -> None:
    agent = {
        "kind": "generation",
        "model": "m",
        "description": "Only modes",
        "modes": {"x": {"instructions": "a.md"}},
    }
    spec = load_folder(write_folder(tmp_path, {"a": agent}, files=FILES))["a"]
    assert spec.needs_mode and spec.body == "" and spec.description == "Only modes"
    assert spec.modes["x"].body == "Work." and not spec.modes["x"].needs_mode


@pytest.mark.parametrize("on_mode", [False, True], ids=["entry", "mode"])
@pytest.mark.parametrize("problem", ["missing", "not utf-8", "a folder"])
def test_instructions_that_cannot_be_read_are_refused(
    tmp_path: Path, problem: str, on_mode: bool
) -> None:
    agent: dict[str, Any] = {**GENERATION, "instructions": "ok.md"}
    who = "agent 'a'"
    if on_mode:
        agent["modes"] = {"x": {"instructions": "b.md"}}
        who = "mode 'x' of agent 'a'"
    else:
        agent["instructions"] = "b.md"
    folder = write_folder(tmp_path, {"a": agent}, files={"ok.md": "Fine.\n"})
    if problem == "not utf-8":
        (folder / "b.md").write_bytes(b"\xff\xfe\xff")
    if problem == "a folder":
        (folder / "b.md").mkdir()
    _refused(folder, rf"^compiled: {who} .*'b\.md'")


def test_editing_another_entry_leaves_the_fingerprint_alone(tmp_path: Path) -> None:
    def write(other: str, changed: str) -> Path:
        agent = {
            **GENERATION,
            "modes": {"x": {"description": changed}, "y": {"description": "Stays"}},
        }
        return write_folder(
            tmp_path,
            {"a": agent, "b": {**GENERATION, "description": other}},
            files={"a.md": "Work.\n"},
        )

    before = load_folder(write("before", "before"))
    after = load_folder(write("after", "after"))
    assert after["a"].fingerprint == before["a"].fingerprint
    assert after["a"].modes["y"].fingerprint == before["a"].modes["y"].fingerprint
    assert after["a"].modes["x"].fingerprint != before["a"].modes["x"].fingerprint
    assert after["b"].fingerprint != before["b"].fingerprint


@pytest.mark.parametrize(
    "edit",
    [
        {"model": "n"},
        {"description": "New"},
        {"max_model_calls": 9},
        {"workspace": True},
        {"files": {"include": ["k"]}},
        {"tools": {"timeout": 3.0}},
        {"skills": {"include": ["s"]}},
        {"orchestration": {"delegate_to": ["b"]}},
    ],
    ids=lambda edit: next(iter(edit)),
)
def test_editing_this_entry_changes_it(tmp_path: Path, edit: dict[str, Any]) -> None:
    written = {**FILES, "k/x.md": "Kept.\n"}
    folder = write_folder(tmp_path, {"a": GENERATION, "b": GENERATION}, files=written)
    before = load_folder(folder)
    write_folder(folder, {"a": {**GENERATION, **edit}, "b": GENERATION}, files=written)
    after = load_folder(folder)
    assert after["a"].fingerprint != before["a"].fingerprint
    assert after["b"].fingerprint == before["b"].fingerprint


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("a.md", "Other.\n"),
        ("q/urgent.json", URGENT.replace("urgent", "pressing")),
        ("q/filled.json", '{"type": "choice", "instructions": "Which one?"}'),
        ("q/filled.py", "def run(input, call):\n    return {'options': {'b': 'B'}}\n"),
        ("q/_shared.py", "NAME = 'other'\n"),
    ],
    ids=["instructions", "question", "scripted question", "script", "helper"],
)
def test_every_file_an_agent_reaches_is_in_its_fingerprint(
    tmp_path: Path, path: str, text: str
) -> None:
    folder = write_folder(
        tmp_path,
        {"t": {**DECISION, "questions": ["q/*.json"]}},
        files={
            "a.md": "Sort.\n",
            "q/urgent.json": URGENT,
            "q/filled.json": '{"type": "choice", "instructions": "Which?"}',
            "q/filled.py": "def run(input, call):\n    return {'options': {'a': 'A'}}\n",
            "q/_shared.py": "NAME = 'one'\n",
        },
    )
    before = load_folder(folder)["t"].fingerprint
    (folder / path).write_text(text, encoding="utf-8")
    assert load_folder(folder)["t"].fingerprint != before


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("x.md", "Other.\n"),
        ("r/area.json", AREA.replace("Which area?", "What area?")),
        ("r/order.json", '{"type": "choice", "instructions": "Which one?"}'),
        ("r/order.py", "def run(input, call):\n    return {'options': {'b': 'B'}}\n"),
        ("r/_own.py", "NAME = 'other'\n"),
    ],
    ids=["instructions", "question", "scripted question", "script", "helper"],
)
def test_a_file_reached_only_through_a_mode_changes_only_that_mode(
    tmp_path: Path, path: str, text: str
) -> None:
    agent = {
        **DECISION,
        "questions": ["q/*.json"],
        "modes": {
            "x": {"instructions": "x.md", "questions": ["r/*.json"]},
            "y": {"description": "Stays"},
        },
    }
    folder = write_folder(
        tmp_path,
        {"t": agent},
        files={
            "a.md": "Sort.\n",
            "x.md": "Sort by mode.\n",
            "q/urgent.json": URGENT,
            "r/area.json": AREA,
            "r/order.json": '{"type": "choice", "instructions": "Which?"}',
            "r/order.py": "def run(input, call):\n    return {'options': {'a': 'A'}}\n",
            "r/_own.py": "NAME = 'one'\n",
        },
    )
    before = load_folder(folder)["t"]
    (folder / path).write_text(text, encoding="utf-8")
    after = load_folder(folder)["t"]
    assert after.modes["x"].fingerprint != before.modes["x"].fingerprint
    assert after.fingerprint == before.fingerprint
    assert after.modes["y"].fingerprint == before.modes["y"].fingerprint


@pytest.mark.parametrize("path", ["notes/other.md", "q/unasked.json", "b.md"])
def test_a_file_it_does_not_reach_leaves_the_fingerprint_alone(tmp_path: Path, path: str) -> None:
    folder = write_folder(
        tmp_path,
        {
            "t": {**DECISION, "questions": ["q/urgent.json"]},
            "b": {**GENERATION, "instructions": "b.md"},
        },
        files={
            "a.md": "Sort.\n",
            "b.md": "B.\n",
            "q/urgent.json": URGENT,
            "q/unasked.json": AREA,
            "notes/other.md": "x\n",
        },
    )
    before = load_folder(folder)["t"].fingerprint
    (folder / path).write_text('{"type": "check", "instructions": "Other?"}', encoding="utf-8")
    assert load_folder(folder)["t"].fingerprint == before
