import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import JsonValue

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Text
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.folder import load_tools, project_of
from rfnry_engine.tree import load_tree

DEF = '{"description": "Do it", "input_schema": {"type": "object"}}'
RUN = "def run(args, call):\n    return {}\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def tool(
    folder: Path,
    name: str,
    description: str = "Do it",
    body: str = RUN,
    input_schema: dict[str, JsonValue] | None = None,
) -> None:
    schema = {"type": "object"} if input_schema is None else input_schema
    write(folder / f"{name}.json", json.dumps({"description": description, "input_schema": schema}))
    write(folder / f"{name}.py", body)


def test_an_agent_sees_the_tools_of_its_folder_and_every_folder_above(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "notify")
    tool(tmp_path / "support/tools", "lookup_order")
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    tool(tmp_path / "support/billing/tools", "refund")
    write(tmp_path / "support/triage/AGENT.md", agent_md("triage"))
    tree = load_tree(tmp_path)
    assert list(tree["billing"].folder_tools) == ["lookup_order", "notify", "refund"]
    assert list(tree["triage"].folder_tools) == ["lookup_order", "notify"]


def test_the_nearest_folder_wins_a_name_clash(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "lookup_order", description="The team's")
    write(tmp_path / "billing/AGENT.md", agent_md("billing"))
    tool(tmp_path / "billing/tools", "lookup_order", description="Billing's own")
    assert (
        load_tree(tmp_path)["billing"].folder_tools["lookup_order"].description == "Billing's own"
    )


def test_a_tool_holds_its_definition_its_script_and_the_folder_that_holds_them(
    tmp_path: Path,
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    schema: dict[str, JsonValue] = {"type": "object", "required": ["order_id"]}
    write(
        tmp_path / "a/tools/fetch.json",
        json.dumps(
            {"description": "Fetch", "input_schema": schema, "output_schema": {"type": "object"}}
        ),
    )
    write(tmp_path / "a/tools/fetch.py", RUN)
    fetch = load_tree(tmp_path)["a"].folder_tools["fetch"]
    assert (fetch.name, fetch.description) == ("fetch", "Fetch")
    assert (fetch.input_schema, fetch.output_schema) == (schema, {"type": "object"})
    assert (fetch.script, fetch.folder) == (tmp_path / "a/tools/fetch.py", tmp_path / "a/tools")


@pytest.mark.parametrize(
    ("files", "said"),
    [
        ({"x.json": DEF}, "has no x.py beside it"),
        ({"x.py": RUN}, "has no x.json beside it"),
        (
            {"x.json": '{"description": "D", "inputSchema": {}}', "x.py": RUN},
            "unknown key 'inputSchema'",
        ),
        (
            {"x.json": '{"description": "D", "input_schema": {"type": "strng"}}', "x.py": RUN},
            "input_schema that is not a valid JSON Schema",
        ),
        ({"x.json": DEF, "x.py": "import rfnry_engine\n" + RUN}, "imports 'rfnry_engine'"),
        ({"x.json": DEF, "x.py": "def run(args, call:\n"}, "x.py"),
    ],
)
def test_a_tool_folder_mistake_is_refused_naming_the_file(
    tmp_path: Path, files: dict[str, str], said: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    for name, text in files.items():
        write(tmp_path / "a/tools" / name, text)
    with pytest.raises(ConfigurationError, match=re.escape(said)):
        load_tree(tmp_path)


def test_each_pinned_tool_folder_message_names_its_file_and_its_fix(tmp_path: Path) -> None:
    folder = tmp_path / "a/tools"
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(folder / "x.json", DEF)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {folder / 'x.json'} has no x.py beside it; a tool is x.json and x.py in a tools/ folder"
    )
    (folder / "x.json").unlink()
    write(folder / "x.py", RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {folder / 'x.py'} has no x.json beside it; describe the tool in x.json, or name a helper _x.py"
    )
    write(folder / "x.json", '{"description": "D", "input_schema": {}, "inputSchema": {}}')
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {folder / 'x.json'} has the unknown key 'inputSchema'; a tool's JSON takes description, input_schema and output_schema"
    )
    write(folder / "x.json", DEF)
    write(folder / "x.py", "from rfnry_engine.models import Text\n" + RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {folder / 'x.py'} imports 'rfnry_engine'; a tool never imports the engine, so keep what it needs in the tool or a helper beside it"
    )


@pytest.mark.parametrize("key", ["input_schema", "output_schema"])
def test_a_schema_that_does_not_compile_is_refused_with_its_reason(
    tmp_path: Path, key: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    bad = {"type": "object", "properties": {"n": {"minLength": -1}}}
    schemas = {"input_schema": {"type": "object"}, key: bad}
    write(tmp_path / "a/tools/x.json", json.dumps({"description": "D", **schemas}))
    write(tmp_path / "a/tools/x.py", RUN)
    said = f"tools/folder: {tmp_path / 'a/tools/x.json'} has an {key} that is not a valid JSON Schema: "
    with pytest.raises(
        ConfigurationError, match=rf"^{re.escape(said)}.*minimum.*; fix it in that file$"
    ):
        load_tree(tmp_path)


def test_a_schema_is_compiled_offline(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    tool(
        tmp_path / "a/tools",
        "x",
        input_schema={"type": "object", "$ref": "https://example.invalid/order.json"},
    )
    with pytest.raises(ConfigurationError, match="Retrieval is disabled"):
        load_tree(tmp_path)


@pytest.mark.parametrize("description", ["", "   ", None, 5])
def test_a_tool_without_a_description_in_text_is_refused_naming_the_file(
    tmp_path: Path, description: object
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    definition: dict[str, object] = {"input_schema": {"type": "object"}}
    if description is not None:
        definition["description"] = description
    write(tmp_path / "a/tools/x.json", json.dumps(definition))
    write(tmp_path / "a/tools/x.py", RUN)
    with pytest.raises(
        ConfigurationError, match=re.escape(f"{tmp_path / 'a/tools/x.json'} gives no description")
    ):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    "schema",
    [{"type": "array"}, {"properties": {}}, {"type": ["object"]}],
)
def test_an_input_schema_whose_top_level_is_not_an_object_is_refused_naming_the_file(
    tmp_path: Path, schema: dict[str, JsonValue]
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    tool(tmp_path / "a/tools", "x", input_schema=schema)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"{tmp_path / 'a/tools/x.json'} has an input_schema whose top level is not"
        ),
    ):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("[1, 2]", "is not a JSON object"),
        ("{nope", "is not valid JSON"),
        ('{"description": "D"}', "has no input_schema"),
        ('{"description": "D", "input_schema": "object"}', "has no input_schema"),
        (
            '{"description": "D", "input_schema": {"type": "object"}, "output_schema": "x"}',
            "has an output_schema that is not an object",
        ),
    ],
)
def test_a_malformed_definition_is_refused_naming_the_file(
    tmp_path: Path, text: str, said: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/tools/x.json", text)
    write(tmp_path / "a/tools/x.py", RUN)
    with pytest.raises(
        ConfigurationError, match=re.escape(f"tools/folder: {tmp_path / 'a/tools/x.json'} {said}")
    ):
        load_tree(tmp_path)


def test_a_definition_or_a_script_that_is_not_utf8_is_refused_naming_it(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    tool(tmp_path / "a/tools", "x")
    (tmp_path / "a/tools/x.py").write_bytes(b"def run(args, call):\n    return '\xff'\n")
    with pytest.raises(
        ConfigurationError, match=re.escape(f"{tmp_path / 'a/tools/x.py'} is not UTF-8")
    ):
        load_tree(tmp_path)


def test_a_helper_is_not_a_tool_and_a_script_may_import_it(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/tools/_client.py", "import json\n\nBASE = 'https://acme.test'\n")
    tool(tmp_path / "a/tools", "fetch", body="import _client\nfrom _client import BASE\n" + RUN)
    assert list(load_tree(tmp_path)["a"].folder_tools) == ["fetch"]


def test_a_helper_s_imports_are_checked_too(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/tools/_client.py", "from rfnry_engine import Engine\n")
    tool(tmp_path / "a/tools", "fetch", body="import _client\n" + RUN)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(f"{tmp_path / 'a/tools/_client.py'} imports 'rfnry_engine'"),
    ):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "def run(args, call):\n    import _client\n    return {}\n",
        "async def run(args, call):\n    from _client import BASE\n    return BASE\n",
        "class Client:\n    import _client\n\n" + RUN,
    ],
)
def test_a_helper_imported_inside_a_function_or_class_is_refused(tmp_path: Path, body: str) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/tools/_client.py", "BASE = 'https://acme.test'\n")
    tool(tmp_path / "a/tools", "fetch", body=body)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {tmp_path / 'a/tools/fetch.py'} imports the helper '_client' inside a function or class; import it at the top of the script, since a helper is reachable only while its folder's scripts load"
    )


def test_a_module_imported_inside_a_function_is_checked_too(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    body = "def run(args, call):\n    import json\n    import rfnry_engine\n    return {}\n"
    tool(tmp_path / "a/tools", "fetch", body=body)
    with pytest.raises(ConfigurationError, match="imports 'rfnry_engine'"):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    ("line", "shown"),
    [("from . import _client", "."), ("from ._client import BASE", "._client")],
)
def test_a_relative_import_is_refused_naming_the_file(
    tmp_path: Path, line: str, shown: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/tools/_client.py", "BASE = 'https://acme.test'\n")
    tool(tmp_path / "a/tools", "fetch", body=f"{line}\n" + RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {tmp_path / 'a/tools/fetch.py'} imports {shown!r} relatively; import a helper of its tools/ folder by its name, such as import _client"
    )


def test_a_script_importing_the_server_s_own_code_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path / "pyproject.toml", "[project]\nname = 'myapp'\n")
    write(tmp_path / "myapp/__init__.py", "SECRET = 1\n")
    write(tmp_path / "agents/a/AGENT.md", agent_md("a"))
    tool(tmp_path / "agents/a/tools", "fetch", body="import json\nfrom myapp import SECRET\n" + RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path / "agents")
    assert str(caught.value) == (
        f"tools/folder: {tmp_path / 'agents/a/tools/fetch.py'} imports 'myapp' from {tmp_path / 'myapp/__init__.py'}, the server's own code; a tool imports the standard library, installed packages and helpers of its tools/ folder"
    )
    body = "import json\nimport urllib.request\nimport yaml\nimport not_installed_anywhere\n" + RUN
    tool(tmp_path / "agents/a/tools", "fetch", body=body)
    assert list(load_tree(tmp_path / "agents")["a"].folder_tools) == ["fetch"]


def test_a_folder_of_the_server_without_an_init_is_refused_naming_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path / "pyproject.toml", "")
    write(tmp_path / "pages/home.py", "TITLE = 'Home'\n")
    write(tmp_path / "agents/a/AGENT.md", agent_md("a"))
    tool(tmp_path / "agents/a/tools", "fetch", body="from pages.home import TITLE\n" + RUN)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(f"imports 'pages' from {tmp_path / 'pages'}, the server's own code"),
    ):
        load_tree(tmp_path / "agents")


def test_a_module_in_a_tools_folder_of_the_tree_is_not_the_server_s_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path / "agents")
    write(tmp_path / "pyproject.toml", "")
    tool(tmp_path / "agents/tools", "notify")
    write(tmp_path / "agents/a/AGENT.md", agent_md("a"))
    tool(tmp_path / "agents/a/tools", "fetch", body="import tools\n" + RUN)
    assert list(load_tree(tmp_path / "agents")["a"].folder_tools) == ["fetch", "notify"]


@pytest.mark.parametrize("name", ["tools", "questions"])
def test_a_server_package_named_like_a_declaration_folder_is_the_server_s_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.syspath_prepend(tmp_path / "src")
    write(tmp_path / "pyproject.toml", "[project]\nname = 'app'\n")
    write(tmp_path / f"src/{name}/__init__.py", "SECRET = 1\n")
    write(tmp_path / "agents/a/AGENT.md", agent_md("a"))
    tool(tmp_path / "agents/a/tools", "fetch", body=f"import {name}\n" + RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path / "agents")
    assert str(caught.value) == (
        f"tools/folder: {tmp_path / 'agents/a/tools/fetch.py'} imports {name!r} from {tmp_path / 'src' / name / '__init__.py'}, the server's own code; a tool imports the standard library, installed packages and helpers of its tools/ folder"
    )


def test_a_tools_folder_off_every_agent_s_path_is_the_server_s_code_to_a_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path / "agents/docs")
    write(tmp_path / "pyproject.toml", "")
    write(tmp_path / "agents/docs/tools/__init__.py", "")
    write(tmp_path / "agents/a/AGENT.md", agent_md("a"))
    tool(tmp_path / "agents/a/tools", "fetch", body="import tools\n" + RUN)
    with pytest.raises(ConfigurationError, match="imports 'tools' from .*, the server's own code"):
        load_tree(tmp_path / "agents")


@pytest.mark.parametrize(
    "body",
    [
        "def main(args, call):\n    return {}\n",
        "class Tool:\n    def run(self, args, call):\n        return {}\n",
        "if True:\n    def run(args, call):\n        return {}\n",
        "run = print\n",
    ],
)
def test_a_tool_script_without_a_run_at_its_top_level_is_refused(tmp_path: Path, body: str) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    tool(tmp_path / "a/tools", "fetch", body=body)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {tmp_path / 'a/tools/fetch.py'} has no run at its top level; a tool defines def run(args, call) or async def run(args, call)"
    )


def test_a_helper_needs_no_run_and_an_async_run_is_a_run(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/tools/_client.py", "BASE = 'https://acme.test'\n")
    body = "import _client\nasync def run(args, call):\n    return _client.BASE\n"
    tool(tmp_path / "a/tools", "fetch", body=body)
    assert list(load_tree(tmp_path)["a"].folder_tools) == ["fetch"]


@pytest.mark.parametrize(
    ("body", "awaits"),
    [
        (RUN, False),
        ("async def run(args, call):\n    return {}\n", True),
        (RUN + "async def run(args, call):\n    return {}\n", True),
        ("async def run(args, call):\n    return {}\n" + RUN, False),
        ("class Tool:\n    async def run(self, args, call):\n        return {}\n" + RUN, False),
    ],
)
def test_a_tool_awaits_when_its_last_run_at_the_top_level_is_async(
    tmp_path: Path, body: str, awaits: bool
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    tool(tmp_path / "a/tools", "fetch", body=body)
    assert load_tree(tmp_path)["a"].folder_tools["fetch"].awaits is awaits


def test_a_package_of_the_interpreter_s_site_packages_inside_the_project_passes(
    tmp_path: Path,
) -> None:
    project = project_of(Path(sys.prefix))
    assert Path(yaml.__file__).resolve().is_relative_to(project)
    tool(tmp_path / "tools", "fetch", body="import yaml\n" + RUN)
    folders = {(tmp_path / "tools").resolve()}
    assert list(load_tools(tmp_path / "tools", project, tmp_path, folders)) == ["fetch"]


def test_a_site_packages_folder_of_the_project_that_is_not_the_interpreter_s_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages = tmp_path / "prefix/lib/python3.14/site-packages"
    monkeypatch.syspath_prepend(packages)
    write(tmp_path / "pyproject.toml", "")
    write(packages / "vendored/__init__.py", "")
    write(tmp_path / "agents/a/AGENT.md", agent_md("a"))
    tool(tmp_path / "agents/a/tools", "fetch", body="import vendored\n" + RUN)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"imports 'vendored' from {packages / 'vendored/__init__.py'}, the server's own code"
        ),
    ):
        load_tree(tmp_path / "agents")


def test_the_project_is_the_nearest_folder_with_a_pyproject_toml_else_the_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "agents/support").mkdir(parents=True)
    assert project_of(tmp_path / "agents/support") == (tmp_path / "agents/support").resolve()
    write(tmp_path / "pyproject.toml", "")
    assert project_of(tmp_path / "agents/support") == tmp_path.resolve()
    assert project_of(tmp_path) == tmp_path.resolve()


def test_a_2020_12_keyword_compiles(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"pair": {"type": "array", "prefixItems": [{"type": "string"}]}},
    }
    tool(tmp_path / "a/tools", "pair", input_schema=schema)
    assert load_tree(tmp_path)["a"].folder_tools["pair"].input_schema == schema


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("Read", ""),
        ("Load", ""),
        ("History", ""),
        ("Output", ""),
        ("Delegate", "agents: [helper]\n"),
        ("Shell", "workspace: true\n"),
        ("Jobs", "workspace: true\n"),
    ],
)
def test_a_tool_named_like_an_engine_tool_is_refused_naming_its_file(
    tmp_path: Path, name: str, extra: str
) -> None:
    write(tmp_path / "support/AGENT.md", agent_md("support", extra=extra))
    write(tmp_path / "helper/AGENT.md", agent_md("helper"))
    tool(tmp_path / "support/tools", name)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: {tmp_path / 'support/tools' / name}.json is named like {name!r}, a tool the engine gives agent 'support'; rename the tool"
    )


@pytest.mark.parametrize("name", ["Delegate", "Shell", "Write"])
def test_a_tool_may_take_the_name_of_an_engine_tool_the_agent_is_not_given(
    tmp_path: Path, name: str
) -> None:
    write(tmp_path / "support/AGENT.md", agent_md("support"))
    tool(tmp_path / "support/tools", name)
    assert list(load_tree(tmp_path)["support"].folder_tools) == [name]


async def test_a_tools_folder_leaves_the_map_and_the_preload_and_stays_readable(
    tmp_path: Path,
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", extra="preload: true\n"))
    write(tmp_path / "a/rules.md", "Be kind.\n")
    tool(tmp_path / "a/tools", "fetch", body="BODY = 'fetches'\n" + RUN)
    write(tmp_path / "a/tools/README.md", "How the tools work.\n")
    spec = load_tree(tmp_path)["a"]
    assert spec.listing == "rules.md"
    sent = "".join(part.text for part in spec.preload if isinstance(part, Text))
    assert "Be kind." in sent and "tools/" not in sent and "fetches" not in sent
    surface = EngineTools(Engine(tmp_path, models={}), agent="a", scope="acme")
    assert "fetches" in str(await surface.dispatch("Read", {"path": "tools/fetch.py"}))
    assert "How the tools work." in str(await surface.dispatch("Read", {"path": "tools/README.md"}))


def test_files_beside_the_tools_and_folders_inside_are_not_tools(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    tool(tmp_path / "a/tools", "fetch")
    write(tmp_path / "a/tools/README.md", "How the tools work.\n")
    write(tmp_path / "a/tools/rates.csv", "a,b\n")
    write(tmp_path / "a/tools/.draft.py", "import rfnry_engine\n")
    write(tmp_path / "a/tools/data/codes.json", "[1, 2]\n")
    write(tmp_path / "a/tools/data/convert.py", "import rfnry_engine\n")
    assert list(load_tree(tmp_path)["a"].folder_tools) == ["fetch"]


def test_editing_a_script_above_an_agent_changes_its_fingerprint(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "notify")
    write(tmp_path / "support/tools/_client.py", "BASE = 1\n")
    tool(tmp_path / "support/tools", "lookup_order", body="import _client\n" + RUN)
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    write(tmp_path / "other/AGENT.md", agent_md("other"))
    first = load_tree(tmp_path)
    write(tmp_path / "tools/notify.py", RUN + "\n# louder\n")
    second = load_tree(tmp_path)
    assert second["billing"].fingerprint != first["billing"].fingerprint
    assert second["other"].fingerprint != first["other"].fingerprint
    write(tmp_path / "support/tools/_client.py", "BASE = 2\n")
    third = load_tree(tmp_path)
    assert third["billing"].fingerprint != second["billing"].fingerprint
    assert third["other"].fingerprint == second["other"].fingerprint


def test_a_tree_moved_with_its_tools_keeps_its_fingerprints(tmp_path: Path) -> None:
    for root in (tmp_path / "here", tmp_path / "there"):
        tool(root / "tools", "notify")
        write(root / "support/AGENT.md", agent_md("support"))
    here, there = load_tree(tmp_path / "here"), load_tree(tmp_path / "there")
    assert here["support"].fingerprint == there["support"].fingerprint


def test_a_decision_agent_sees_no_folder_tools(tmp_path: Path) -> None:
    tool(tmp_path / "tools", "notify")
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    write(tmp_path / "triage/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}')
    assert load_tree(tmp_path)["triage"].folder_tools == {}


def test_a_decision_agent_s_folder_may_hold_tools_for_the_agents_inside_it(
    tmp_path: Path,
) -> None:
    write(tmp_path / "support/AGENT.md", agent_md("support", kind="decision", model="judge"))
    write(
        tmp_path / "support/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}'
    )
    tool(tmp_path / "support/tools", "lookup_order")
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    tree = load_tree(tmp_path)
    assert tree["support"].folder_tools == {}
    assert list(tree["billing"].folder_tools) == ["lookup_order"]


def test_a_tools_folder_above_only_decision_agents_is_still_checked(tmp_path: Path) -> None:
    write(tmp_path / "tools/x.json", DEF)
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    write(tmp_path / "triage/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}')
    with pytest.raises(ConfigurationError, match="has no x.py beside it"):
        load_tree(tmp_path)


def test_a_tools_folder_off_every_agent_s_path_is_content(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/docs/tools/x.json", DEF)
    spec = load_tree(tmp_path)["a"]
    assert spec.folder_tools == {}
    assert "x.json" in spec.listing


def test_no_agent_is_found_inside_a_tools_folder(tmp_path: Path) -> None:
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    tool(tmp_path / "support/tools", "lookup_order")
    write(tmp_path / "support/tools/sub/AGENT.md", agent_md("hidden"))
    write(tmp_path / "tools/inner/AGENT.md", agent_md("inner"))
    write(tmp_path / "support/billing/tools/deep/AGENT.md", agent_md("deep"))
    write(tmp_path / "support/billing/docs/tools/x/AGENT.md", agent_md("docs"))
    assert list(load_tree(tmp_path)) == ["billing"]


async def test_a_group_folder_s_tools_leave_the_map_and_preload_of_the_agent_around_it(
    tmp_path: Path,
) -> None:
    write(tmp_path / "AGENT.md", agent_md("lead", extra="preload: true\n"))
    write(tmp_path / "notes.md", "Be kind.\n")
    write(tmp_path / "support/guide.md", "Answer fast.\n")
    tool(tmp_path / "support/tools", "lookup_order", body="BODY = 'looks up'\n" + RUN)
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    tree = load_tree(tmp_path)
    lead = tree["lead"]
    assert lead.listing == "notes.md\nsupport/\n  guide.md"
    sent = "".join(part.text for part in lead.preload if isinstance(part, Text))
    assert "Answer fast." in sent and "lookup_order" not in sent and "looks up" not in sent
    assert lead.folder_tools == {} and list(tree["billing"].folder_tools) == ["lookup_order"]
    surface = EngineTools(Engine(tmp_path, models={}), agent="lead", scope="acme")
    read = await surface.dispatch("Read", {"path": "support/tools/lookup_order.py"})
    assert "looks up" in str(read)


def test_a_read_folder_s_tools_leave_the_reader_s_map_and_preload(tmp_path: Path) -> None:
    write(tmp_path / "support/guide.md", "---\ndescription: Answer fast\n---\nAnswer fast.\n")
    tool(tmp_path / "support/tools", "lookup_order", body="BODY = 'looks up'\n" + RUN)
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    write(tmp_path / "desk/AGENT.md", agent_md("desk", extra="reads: [support]\npreload: true\n"))
    desk = load_tree(tmp_path)["desk"]
    assert desk.listing == "../support/\n  guide.md — Answer fast"
    sent = "".join(part.text for part in desk.preload if isinstance(part, Text))
    assert "Answer fast." in sent and "lookup_order" not in sent and "looks up" not in sent


@pytest.mark.parametrize(
    "name", ["tools", "support/tools", "support/tools/data", "support/billing/tools"]
)
def test_a_tools_folder_on_an_agent_s_path_cannot_be_named_in_reads(
    tmp_path: Path, name: str
) -> None:
    tool(tmp_path / "tools", "notify")
    tool(tmp_path / "support/tools", "lookup_order")
    write(tmp_path / "support/tools/data/codes.json", "[]\n")
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    tool(tmp_path / "support/billing/tools", "refund")
    write(tmp_path / "desk/AGENT.md", agent_md("desk", extra=f"reads: [{name}]\n"))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: agent 'desk' reads {name!r}, a tools/ folder or a folder inside one; a tools/ folder gives its tools to the agents below it and is not read as content, so drop it from reads:"
    )


def test_a_tools_folder_off_every_agent_s_path_can_be_read_as_content(tmp_path: Path) -> None:
    write(tmp_path / "kb/tools/codes.md", "---\ndescription: Codes\n---\nA-1 is open.\n")
    write(tmp_path / "kb/tools/lookup.json", DEF)
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/docs/tools/notes.md", "---\ndescription: Notes\n---\nBe brief.\n")
    write(tmp_path / "b/AGENT.md", agent_md("b", extra="reads: [kb/tools, a/docs/tools]\n"))
    b = load_tree(tmp_path)["b"]
    assert b.listing == (
        "../kb/tools/\n  codes.md — Codes\n  lookup.json\n../a/docs/tools/\n  notes.md — Notes"
    )
    assert b.folder_tools == {}


def test_an_unknown_read_folder_is_offered_every_folder_but_the_tools_folders_on_a_path(
    tmp_path: Path,
) -> None:
    tool(tmp_path / "tools", "notify")
    tool(tmp_path / "support/tools", "lookup_order")
    write(tmp_path / "support/billing/AGENT.md", agent_md("billing"))
    write(tmp_path / "kb/tools/codes.md", "A-1 is open.\n")
    write(tmp_path / "desk/AGENT.md", agent_md("desk", extra="reads: [nowhere]\n"))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value).endswith("one of: desk, kb, kb/tools, support, support/billing")
