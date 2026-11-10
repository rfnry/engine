import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, text_reply
from .trees import write_folder

GENERATION: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "a.md"}
RUN = "def run(args, call):\n    return 'ran'\n"
WAIT = "import asyncio\nasync def run(args, call):\n    await asyncio.sleep(30)\n"
ONE: dict[str, Any] = {"module": "tools/one.py", "description": "Does one"}
OTHER: dict[str, Any] = {"module": "tools/other.py", "description": "Does the other"}
ORDER: dict[str, Any] = {
    "description": "From the schema file",
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "output_schema": {"type": "string"},
}
COUNTED = (
    "from pathlib import Path\n"
    "with Path(__file__).with_name('loads.txt').open('a', encoding='utf-8') as file:\n"
    "    file.write(Path(__file__).stem + '\\n')\n"
)


def _folder(
    root: Path,
    tools: dict[str, Any],
    written: dict[str, str],
    *,
    include: list[str] | None = None,
    section: dict[str, Any] | None = None,
    **agent: Any,
) -> Path:
    picks = {"include": ["*"] if include is None else include, **(section or {})}
    return write_folder(
        root,
        {"a": {**GENERATION, "tools": picks, **agent}},
        tools=tools or None,
        files={"a.md": "Work.\n", **written},
    )


def _refused(folder: Path, pattern: str) -> None:
    with pytest.raises(ConfigurationError, match=pattern):
        load_folder(folder)


def _calling(name: str, **args: Any) -> Reply:
    return Reply(tool_calls=[ToolCall(id="c1", name=name, args=args)], stop_reason="tool_use")


def _offered(model: Scripted) -> list[str]:
    return sorted(tool.name for tool in model.calls[0][2])


async def _answer(folder: Path, name: str, args: Any = None) -> Any:
    async with Engine(folder, models={"m": Scripted()}) as engine:
        surface = EngineTools(engine, agent="a", scope="acme")
        return await surface.dispatch(name, {} if args is None else args)


def _loads(folder: Path) -> list[str]:
    return sorted((folder / "tools/loads.txt").read_text(encoding="utf-8").splitlines())


async def test_the_name_is_the_key_not_the_file(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"find_slots": {"module": "tools/slots_v2.py", "description": "Finds slots"}},
        {"tools/slots_v2.py": "def run(args, call):\n    return call['tool']\n"},
    )
    model = Scripted(_calling("find_slots"), text_reply("done"))
    async with Engine(folder, models={"m": model}) as engine:
        run = await engine.run("a", "acme", "go")
    assert isinstance(run, Generation)
    assert _offered(model) == ["find_slots"]
    assert run.steps[0].tool == "find_slots"
    assert run.steps[0].result == "find_slots"


async def test_the_function_is_named_and_defaults_to_run(tmp_path: Path) -> None:
    source = (
        "def run(args, call):\n    return 'the default'\n"
        "async def other(args, call):\n    return 'the named one'\n"
    )
    folder = _folder(
        tmp_path,
        {
            "plain": {"module": "tools/both.py", "description": "Runs run"},
            "named": {"module": "tools/both.py", "function": "other", "description": "Runs other"},
        },
        {"tools/both.py": source},
    )
    async with Engine(folder, models={"m": Scripted()}) as engine:
        surface = EngineTools(engine, agent="a", scope="acme")
        assert await surface.dispatch("plain", {}) == "the default"
        assert await surface.dispatch("named", {}) == "the named one"


@pytest.mark.parametrize("module", ["tools/one.sh", "tools/one", "tools/one.py.txt"])
def test_a_module_that_is_not_python_is_refused(tmp_path: Path, module: str) -> None:
    folder = _folder(
        tmp_path, {"one": {**ONE, "module": module}}, {module: RUN, "tools/one.py": RUN}
    )
    _refused(folder, r"^tools/folder: tool 'one' has the module .*\.py")


@pytest.mark.parametrize(
    ("entry", "written", "pattern"),
    [
        ({}, {}, r"^tools/folder: tool 'one' names the module 'tools/one\.py'"),
        (
            {"function": "absent"},
            {"tools/one.py": RUN},
            r"^tools/folder: .*one\.py has no absent at its top level; a tool defines def absent\(args, call\)",
        ),
        (
            {"function": "nested"},
            {"tools/one.py": "class Holder:\n    def nested(self):\n        pass\n" + RUN},
            r"^tools/folder: .*one\.py has no nested at its top level",
        ),
    ],
    ids=["module", "function", "not-at-the-top-level"],
)
def test_a_missing_module_or_function_is_refused(
    tmp_path: Path, entry: dict[str, Any], written: dict[str, str], pattern: str
) -> None:
    _refused(_folder(tmp_path, {"one": {**ONE, **entry}}, written), pattern)


def test_a_module_is_checked_without_being_imported(tmp_path: Path) -> None:
    source = "from pathlib import Path\nPath(__file__).with_name('ran').write_text('x')\n" + RUN
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    load_folder(folder)
    assert not (folder / "tools/ran").exists()


async def test_without_a_schema_any_object_goes_in_and_the_result_is_unchecked(
    tmp_path: Path,
) -> None:
    source = "def run(args, call):\n    return [args, 1, 'two']\n"
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    tool = load_folder(folder)["a"].folder_tools["one"]
    assert (tool.input_schema, tool.output_schema) == ({"type": "object"}, None)
    given = {"anything": {"at": ["all"]}}
    assert await _answer(folder, "one", given) == [given, 1, "two"]
    with pytest.raises(InputError, match=r"^tools/declared: invalid arguments for one: "):
        await _answer(folder, "one", ["not", "an", "object"])


def test_the_description_comes_from_the_entry_else_the_schema_file(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {
            "written": {
                "module": "tools/one.py",
                "description": "From the entry",
                "schema": "tools/order.json",
            },
            "from_file": {"module": "tools/one.py", "schema": "tools/order.json"},
        },
        {"tools/one.py": RUN, "tools/order.json": json.dumps(ORDER)},
    )
    tools = load_folder(folder)["a"].folder_tools
    assert tools["written"].description == "From the entry"
    assert tools["from_file"].description == "From the schema file"
    assert tools["from_file"].input_schema == ORDER["input_schema"]
    assert tools["from_file"].output_schema == ORDER["output_schema"]


@pytest.mark.parametrize(
    ("entry", "written", "pattern"),
    [
        ({}, {}, r"^tools/folder: tool 'one' has no description"),
        (
            {"schema": "tools/one.json"},
            {"tools/one.json": json.dumps({"input_schema": {"type": "object"}})},
            r"^tools/folder: .*one\.json gives no description in text",
        ),
        (
            {"schema": "tools/one.json"},
            {
                "tools/one.json": json.dumps(
                    {"description": " ", "input_schema": {"type": "object"}}
                )
            },
            r"^tools/folder: .*one\.json gives no description in text",
        ),
    ],
    ids=["no-schema", "schema-without-one", "schema-with-a-blank-one"],
)
def test_no_description_anywhere_is_refused(
    tmp_path: Path, entry: dict[str, Any], written: dict[str, str], pattern: str
) -> None:
    tools = {"one": {"module": "tools/one.py", **entry}}
    _refused(_folder(tmp_path, tools, {"tools/one.py": RUN, **written}), pattern)


def test_a_schema_file_takes_only_its_three_keys(tmp_path: Path) -> None:
    extra = {**ORDER, "name": "not_this"}
    folder = _folder(
        tmp_path,
        {"one": {"module": "tools/one.py", "schema": "tools/one.json"}},
        {"tools/one.py": RUN, "tools/one.json": json.dumps(extra)},
    )
    _refused(
        folder,
        r"^tools/folder: .*one\.json has the unknown key 'name'; a tool's JSON takes description, input_schema and output_schema",
    )


def test_a_schema_file_that_is_missing_is_refused(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"one": {**ONE, "schema": "tools/absent.json"}},
        {"tools/one.py": RUN},
    )
    _refused(folder, r"^tools/folder: tool 'one' names the schema 'tools/absent\.json'")


@pytest.mark.parametrize(
    ("own", "agent", "engine"),
    [
        (0.2, 30.0, 30.0),
        (30.0, 0.2, 30.0),
        (30.0, 30.0, 0.2),
        (None, None, 0.2),
        (0.2, None, None),
    ],
    ids=["the-tool", "the-agent", "the-engine", "only-the-engine", "only-the-tool"],
)
async def test_the_smallest_timeout_wins(
    tmp_path: Path, own: float | None, agent: float | None, engine: float | None
) -> None:
    entry: dict[str, Any] = {"module": "tools/patient.py", "description": "Waits"}
    if own is not None:
        entry["timeout"] = own
    section = {} if agent is None else {"timeout": agent}
    folder = _folder(tmp_path, {"wait": entry}, {"tools/patient.py": WAIT}, section=section)
    model = Scripted(_calling("wait"), text_reply("done"))
    async with Engine(folder, models={"m": model}, tool_timeout=engine) as held:
        run = await held.run("a", "acme", "go")
    assert isinstance(run, Generation)
    assert run.steps[0].is_error
    assert run.steps[0].result == "TimeoutError: workers/pool: wait timed out after 0.2s"


async def test_the_agents_timeout_also_bounds_a_tool_the_run_gives(tmp_path: Path) -> None:
    async def patient(args: dict[str, Any]) -> str:
        await asyncio.sleep(30)
        return "late"

    folder = _folder(tmp_path, {}, {}, include=["given"], section={"timeout": 0.2})
    given = Tool("given", "Waits", body=patient)
    model = Scripted(_calling("given"), text_reply("done"))
    async with Engine(folder, models={"m": model}, tool_timeout=30.0) as engine:
        run = await engine.run("a", "acme", "go", tools=Tools([given]))
    assert isinstance(run, Generation)
    assert run.steps[0].result == "TimeoutError: engine: given timed out after 0.2s"


async def test_a_name_is_declared_then_built_in_then_from_the_run(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"one": ONE},
        {"tools/one.py": RUN},
        include=["one", "Grep", "given"],
    )
    spec = load_folder(folder)["a"]
    assert list(spec.folder_tools) == ["one"]
    assert spec.keep == ["Grep", "given", "one"]
    model = Scripted(text_reply("done"))
    given = Tool("given", "A tool of the run", body=None)
    async with Engine(folder, models={"m": model}) as engine:
        await engine.run("a", "acme", "go", tools=Tools([given]))
    assert _offered(model) == ["Grep", "given", "one"]


async def test_a_pattern_matches_declared_names_only(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"find_a": ONE, "find_b": ONE, "Grepper": ONE},
        {"tools/one.py": RUN},
        include=["find_*", "*per"],
    )
    model = Scripted(text_reply("done"))
    given = Tool("find_given", "A tool of the run", body=None)
    async with Engine(folder, models={"m": model}) as engine:
        await engine.run("a", "acme", "go", tools=Tools([given]))
    assert _offered(model) == ["Grepper", "find_a", "find_b"]


async def test_a_star_is_the_only_wildcard_of_a_pattern(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {**GENERATION, "tools": {"include": ["find_[ab]*"]}},
            "b": {**GENERATION, "tools": {"include": ["find_a"]}},
        },
        tools={"find_[ab]x": ONE, "find_a": ONE},
        files={"a.md": "Work.\n", "tools/one.py": RUN},
    )
    assert list(load_folder(folder)["a"].folder_tools) == ["find_[ab]x"]


def test_exclude_removes_one_of_a_pattern(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **GENERATION,
                "tools": {"include": ["*"], "exclude": ["kept_out", "dropped_*"]},
                "modes": {
                    "chat": {"tools": {"include": ["kept_out"], "exclude": ["kept_in"]}},
                },
            },
            "b": {**GENERATION, "tools": {"include": ["kept_out", "dropped_a"]}},
        },
        tools={"kept_in": ONE, "stays": ONE, "kept_out": ONE, "dropped_a": ONE},
        files={"a.md": "Work.\n", "tools/one.py": RUN},
    )
    spec = load_folder(folder)["a"]
    assert sorted(spec.folder_tools) == ["kept_in", "stays"]
    assert sorted(spec.modes["chat"].folder_tools) == ["stays"]
    assert spec.modes["chat"].keep == ["stays"]


def test_excluding_a_built_in_drops_it_and_a_pattern_never_does(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **GENERATION,
                "files": {"include": ["k.md"]},
                "tools": {"include": ["one"], "exclude": ["Grep"]},
            },
            "b": {
                **GENERATION,
                "files": {"include": ["k.md"]},
                "tools": {"include": ["one", "other"], "exclude": ["o*r"]},
            },
        },
        tools={"one": ONE, "other": ONE},
        files={"a.md": "Work.\n", "k.md": "Known.\n", "tools/one.py": RUN},
    )
    with pytest.raises(
        ConfigurationError, match=r"tools\.other in rfnry\.yaml is picked by nothing"
    ):
        load_folder(folder)
    write_folder(
        folder,
        {
            "a": {
                **GENERATION,
                "files": {"include": ["k.md"]},
                "tools": {"include": ["one", "other"], "exclude": ["Grep", "o*r"]},
            },
            "b": {**GENERATION, "tools": {"include": ["other"]}},
        },
        tools={"one": ONE, "other": ONE},
    )
    agents = load_folder(folder)
    assert agents["a"].keep == ["Glob", "LS", "Load", "Read", "one"]


@pytest.mark.parametrize("key", ["include", "exclude"])
def test_a_pattern_matching_nothing_is_refused(tmp_path: Path, key: str) -> None:
    section = {"include": ["one"], key: ["one", "nothing_*"] if key == "include" else ["nothing_*"]}
    folder = write_folder(
        tmp_path,
        {"a": {**GENERATION, "tools": section}},
        tools={"one": ONE},
        files={"a.md": "Work.\n", "tools/one.py": RUN},
    )
    _refused(
        folder,
        rf"^compiled: agent 'a' lists the pattern 'nothing_\*' in tools\.{key}, which matches no name under tools:",
    )


def test_a_pattern_on_a_mode_matching_nothing_names_the_mode(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"one": ONE},
        {"tools/one.py": RUN},
        modes={"chat": {"tools": {"include": ["nothing_*"]}}},
    )
    _refused(
        folder,
        r"^compiled: mode 'chat' of agent 'a' lists the pattern 'nothing_\*' in tools\.include",
    )


@pytest.mark.parametrize(
    "section",
    [{"include": ["one"]}, {"include": ["*"], "exclude": ["loose"]}],
    ids=["left-out", "taken-back"],
)
def test_a_declared_tool_nothing_picks_is_refused(tmp_path: Path, section: dict[str, Any]) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**GENERATION, "tools": section}},
        tools={"one": ONE, "loose": OTHER},
        files={"a.md": "Work.\n", "tools/one.py": RUN, "tools/other.py": RUN},
    )
    _refused(folder, r"^compiled: tools\.loose in rfnry\.yaml is picked by nothing")


def test_a_tool_only_a_mode_or_another_agent_picks_counts_as_picked(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **GENERATION,
                "tools": {"include": ["one"]},
                "modes": {"chat": {"tools": {"include": ["by_a_mode"]}}},
            },
            "b": {**GENERATION, "tools": {"include": ["by_another"]}},
        },
        tools={"one": ONE, "by_a_mode": ONE, "by_another": ONE},
        files={"a.md": "Work.\n", "tools/one.py": RUN},
    )
    agents = load_folder(folder)
    assert sorted(agents["a"].folder_tools) == ["one"]
    assert sorted(agents["a"].modes["chat"].folder_tools) == ["by_a_mode", "one"]
    assert sorted(agents["b"].folder_tools) == ["by_another"]


@pytest.mark.parametrize(
    "name", ["Read", "Grep", "Glob", "LS", "Load", "Shell", "Jobs", "Delegate", "History", "Output"]
)
def test_a_declared_tool_named_like_a_built_in_is_refused(tmp_path: Path, name: str) -> None:
    folder = _folder(tmp_path, {name: ONE}, {"tools/one.py": RUN})
    _refused(
        folder, rf"^compiled: tools\.{name} in rfnry\.yaml is named like a tool the engine gives"
    )


async def test_a_runs_tool_replaces_a_declared_tool_of_the_same_name(tmp_path: Path) -> None:
    async def stub(args: dict[str, Any]) -> str:
        return "the stub"

    source = "def run(args, call):\n    raise RuntimeError('the real system was called')\n"
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    model = Scripted(_calling("one"), text_reply("done"))
    given = Tool("one", "A stand-in", body=stub)
    async with Engine(folder, models={"m": model}) as engine:
        run = await engine.run("a", "acme", "go", tools=Tools([given]))
    assert isinstance(run, Generation)
    assert run.steps[0].result == "the stub"
    assert [tool.description for tool in model.calls[0][2]] == ["A stand-in"]


async def test_a_bare_name_the_run_does_not_give_fails_before_any_model_call(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, {}, {}, include=["absent"])
    model = Scripted(text_reply("done"))
    async with Engine(folder, models={"m": model}) as engine:
        with pytest.raises(ConfigurationError, match=r"keeps 'absent' in tools:"):
            await engine.run("a", "acme", "go")
    assert model.calls == []


async def test_a_run_tool_no_entry_names_stays_out(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": RUN})
    model = Scripted(text_reply("done"))
    given = Tool("given", "A tool of the run", body=None)
    async with Engine(folder, models={"m": model}) as engine:
        await engine.run("a", "acme", "go", tools=Tools([given]))
    assert _offered(model) == ["one"]


async def test_a_tool_reaches_only_modules_beside_it(tmp_path: Path) -> None:
    uses = "import shared\ndef run(args, call):\n    return shared.VALUE\n"
    parent = tmp_path / "parent"
    parent.mkdir()
    _refused(
        _folder(
            parent,
            {"one": {**ONE, "module": "tools/deep/one.py"}},
            {"tools/deep/one.py": uses, "tools/shared.py": "VALUE = 'above'\n"},
        ),
        r"^tools/folder: .*one\.py imports 'shared', which resolves to nothing; a tool reaches the standard library, installed packages and the single \.py modules in its own folder",
    )
    beside = tmp_path / "beside"
    beside.mkdir()
    folder = _folder(
        beside,
        {"one": ONE},
        {
            "tools/one.py": uses,
            "tools/shared.py": "import deeper\nVALUE = deeper.VALUE\n",
            "tools/deeper.py": "VALUE = 'beside'\n",
        },
    )
    assert sorted(load_folder(folder)["a"].folder_tools["one"].helpers) == ["deeper", "shared"]
    assert await _answer(folder, "one") == "beside"


def test_a_folder_beside_a_tool_is_not_a_module_it_reaches(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"one": ONE},
        {
            "tools/one.py": "import shared\n" + RUN,
            "tools/shared/__init__.py": "VALUE = 1\n",
        },
    )
    _refused(folder, r"imports 'shared', which resolves to nothing.*single \.py modules")


async def test_two_tools_of_a_folder_each_reach_their_own_modules(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"one": ONE, "other": OTHER},
        {
            "tools/one.py": "import for_one\ndef run(args, call):\n    return for_one.VALUE\n",
            "tools/other.py": "import for_other\ndef run(args, call):\n    return for_other.VALUE\n",
            "tools/for_one.py": "VALUE = 'one'\n",
            "tools/for_other.py": "VALUE = 'other'\n",
        },
    )
    async with Engine(folder, models={"m": Scripted()}) as engine:
        surface = EngineTools(engine, agent="a", scope="acme")
        assert await surface.dispatch("one", {}) == "one"
        assert await surface.dispatch("other", {}) == "other"


async def test_a_module_two_tools_share_loads_once(tmp_path: Path) -> None:
    both = (
        COUNTED
        + "import counted_helper\n"
        + "def run(args, call):\n    return 'first'\n"
        + "def second(args, call):\n    return 'second'\n"
    )
    folder = _folder(
        tmp_path,
        {
            "first": {"module": "tools/both.py", "description": "First"},
            "second": {"module": "tools/both.py", "function": "second", "description": "Second"},
            "third": {"module": "tools/third.py", "description": "Third"},
        },
        {
            "tools/both.py": both,
            "tools/third.py": "import counted_helper\ndef run(args, call):\n    return 'third'\n",
            "tools/counted_helper.py": COUNTED,
        },
    )
    async with Engine(folder, models={"m": Scripted()}) as engine:
        surface = EngineTools(engine, agent="a", scope="acme")
        assert [await surface.dispatch(name, {}) for name in ("first", "second", "third")] == [
            "first",
            "second",
            "third",
        ]
    assert _loads(folder) == ["both", "counted_helper"]


async def test_a_call_holds_a_thread_only_when_its_function_takes_one(tmp_path: Path) -> None:
    source = (
        "def run(args, call):\n    return 'threaded'\n"
        "async def other(args, call):\n    return 'on the loop'\n"
    )
    folder = _folder(
        tmp_path,
        {
            "plain": {"module": "tools/both.py", "description": "Runs run"},
            "named": {"module": "tools/both.py", "function": "other", "description": "Runs other"},
        },
        {"tools/both.py": source},
    )
    async with Engine(folder, models={"m": Scripted()}) as engine:
        await EngineTools(engine, agent="a", scope="acme").dispatch("plain", {})
        [worker] = engine._workers.pool("a")._workers
        assert worker is not None and worker.threaded == {"tools/both:run"}


@pytest.mark.parametrize(
    "imports",
    [
        "import not_installed_anywhere\n",
        "from not_installed_anywhere.deep import thing\n",
        "def later():\n    import not_installed_anywhere\n",
        "try:\n    pass\nexcept ImportError:\n    import not_installed_anywhere\n",
        "try:\n    import not_installed_anywhere\nexcept ValueError:\n    pass\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    import not_installed_anywhere\n",
    ],
    ids=["import", "from", "in-a-function", "in-the-handler", "other-error", "else-of-the-guard"],
)
def test_an_import_that_resolves_to_nothing_is_refused(tmp_path: Path, imports: str) -> None:
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": imports + RUN})
    _refused(
        folder,
        r"^tools/folder: .*one\.py imports 'not_installed_anywhere', which resolves to nothing; ",
    )


async def test_the_standard_library_and_installed_packages_resolve(tmp_path: Path) -> None:
    source = (
        "import sys, os, json\n"
        "import xml.etree.ElementTree\n"
        "import yaml\n"
        "def run(args, call):\n"
        "    return [sys.platform != '', os.sep, json.dumps(1), yaml.safe_load('2')]\n"
    )
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    assert await _answer(folder, "one") == [True, "/", "1", 2]


@pytest.mark.parametrize(
    "caught",
    ["ImportError", "ModuleNotFoundError", "(ValueError, ImportError)"],
)
async def test_an_import_under_try_except_importerror_loads(tmp_path: Path, caught: str) -> None:
    source = (
        "try:\n"
        "    import not_installed_anywhere as optional\n"
        f"except {caught}:\n"
        "    optional = None\n"
        "def run(args, call):\n"
        "    return optional is None\n"
    )
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    assert await _answer(folder, "one") is True


@pytest.mark.parametrize(
    ("imported", "guard"),
    [
        ("from typing import TYPE_CHECKING\n", "TYPE_CHECKING"),
        ("import typing\n", "typing.TYPE_CHECKING"),
    ],
    ids=["name", "attribute"],
)
async def test_an_import_under_type_checking_loads(
    tmp_path: Path, imported: str, guard: str
) -> None:
    source = f"{imported}if {guard}:\n    import not_installed_anywhere\n" + RUN
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    assert await _answer(folder, "one") == "ran"


@pytest.mark.parametrize(
    ("source", "pattern"),
    [
        ("import rfnry_engine\n" + RUN, r"imports 'rfnry_engine'; a tool never imports the engine"),
        ("from . import shared\n" + RUN, r"imports '\.' relatively; import a module beside it"),
        (
            "def later():\n    import shared\n" + RUN,
            r"imports the helper 'shared' inside a function or class",
        ),
    ],
    ids=["the-engine", "relative", "a-module-beside-it-imported-late"],
)
def test_a_tool_keeps_the_rules_of_its_imports(tmp_path: Path, source: str, pattern: str) -> None:
    folder = _folder(
        tmp_path, {"one": ONE}, {"tools/one.py": source, "tools/shared.py": "VALUE = 1\n"}
    )
    _refused(folder, rf"^tools/folder: .*one\.py {pattern}")


def test_a_module_beside_it_named_like_the_standard_library_is_refused(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"one": ONE},
        {"tools/one.py": "import json\n" + RUN, "tools/json.py": "VALUE = 1\n"},
    )
    _refused(folder, r"^tools/folder: .*json\.py is named like the standard-library module 'json'")


def test_a_module_or_its_neighbour_linking_out_of_the_root_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text(RUN, encoding="utf-8")
    linked = tmp_path / "linked"
    linked.mkdir()
    folder = _folder(linked, {"one": ONE}, {"a.md": "Work.\n"})
    (folder / "tools").mkdir()
    (folder / "tools/one.py").symlink_to(outside)
    _refused(folder, r"^tools/folder: .*one\.py links to .*outside\.py, outside the folder root")
    (folder / "tools/one.py").unlink()
    (folder / "tools/one.py").write_text("import shared\n" + RUN, encoding="utf-8")
    (folder / "tools/shared.py").symlink_to(outside)
    _refused(folder, r"^tools/folder: .*shared\.py links to .*outside\.py, outside the folder root")


async def test_a_file_nothing_names_is_not_a_tool(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": RUN})
    before = load_folder(folder)["a"].fingerprint
    for name in ("x_test.py", "conftest.py"):
        (folder / "tools" / name).write_text(
            "import not_installed_anywhere\nimport pytest\nraise SystemExit('it ran')\n",
            encoding="utf-8",
        )
    spec = load_folder(folder)["a"]
    assert list(spec.folder_tools) == ["one"]
    assert spec.fingerprint == before
    async with Engine(folder, models={"m": Scripted()}) as engine:
        held = cast(Any, engine._workers.pool("a")._scripts)
        assert [script["key"] for script in held["scripts"]] == ["tools/one"]
        assert held["helpers"] == {str(folder.resolve() / "tools"): {}}
        assert await EngineTools(engine, agent="a", scope="acme").dispatch("one", {}) == "ran"


async def test_a_tool_with_a_footer_runs_in_the_worker(tmp_path: Path) -> None:
    source = RUN + "\nif __name__ == '__main__':\n    raise SystemExit('the footer ran')\n"
    folder = _folder(tmp_path, {"one": ONE}, {"tools/one.py": source})
    assert await _answer(folder, "one") == "ran"


def test_a_tools_own_files_leave_its_map_and_the_rest_stays(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **GENERATION,
                "files": {"include": ["tools"]},
                "tools": {"include": ["one"]},
            },
            "b": {**GENERATION, "tools": {"include": ["other"]}},
        },
        tools={
            "one": {"module": "tools/one.py", "schema": "tools/one.json"},
            "other": OTHER,
        },
        files={
            "a.md": "Work.\n",
            "tools/one.py": "import shared\n" + RUN,
            "tools/one.json": json.dumps(ORDER),
            "tools/shared.py": "VALUE = 1\n",
            "tools/other.py": RUN,
            "tools/notes.md": "Notes.\n",
        },
    )
    assert load_folder(folder)["a"].listing.splitlines() == ["tools/", "  notes.md", "  other.py"]


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("tools/one.py", "import shared\n" + RUN + "VALUE = 2\n"),
        ("tools/shared.py", "VALUE = 2\n"),
        ("tools/one.json", json.dumps({**ORDER, "description": "Edited"})),
    ],
    ids=["module", "helper", "schema"],
)
def test_what_a_tool_is_made_of_is_in_the_fingerprint(tmp_path: Path, path: str, text: str) -> None:
    folder = _folder(
        tmp_path,
        {"one": {"module": "tools/one.py", "schema": "tools/one.json"}},
        {
            "tools/one.py": "import shared\n" + RUN,
            "tools/one.json": json.dumps(ORDER),
            "tools/shared.py": "VALUE = 1\n",
        },
    )
    before = load_folder(folder)["a"].fingerprint
    (folder / path).write_text(text, encoding="utf-8")
    assert load_folder(folder)["a"].fingerprint != before


@pytest.mark.parametrize(
    "edit", [{"function": "other"}, {"timeout": 5.0}, {"description": "Edited"}]
)
def test_a_tools_entry_is_in_the_fingerprint(tmp_path: Path, edit: dict[str, Any]) -> None:
    source = RUN + "def other(args, call):\n    return 1\n"
    written = {"tools/one.py": source}
    before = load_folder(_folder(tmp_path, {"one": ONE}, written))["a"].fingerprint
    after = load_folder(_folder(tmp_path, {"one": {**ONE, **edit}}, written))["a"].fingerprint
    assert after != before


def test_a_tool_the_agent_does_not_pick_leaves_its_fingerprint_alone(tmp_path: Path) -> None:
    agents = {
        "a": {**GENERATION, "tools": {"include": ["one"]}},
        "b": {**GENERATION, "tools": {"include": ["other"]}},
    }
    files = {"a.md": "Work.\n", "tools/one.py": RUN, "tools/other.py": RUN}
    folder = write_folder(tmp_path, agents, tools={"one": ONE, "other": OTHER}, files=files)
    before = load_folder(folder)["a"].fingerprint
    (folder / "tools/other.py").write_text(RUN + "VALUE = 2\n", encoding="utf-8")
    write_folder(folder, agents, tools={"one": ONE, "other": {**OTHER, "timeout": 5.0}})
    assert load_folder(folder)["a"].fingerprint == before
