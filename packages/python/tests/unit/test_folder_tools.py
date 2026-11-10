import json
import re
import site
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from pydantic import JsonValue

from rfnry_engine.compiled import load_folder
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.manifest import ToolEntry
from rfnry_engine.tools import folder
from rfnry_engine.tools.folder import load_tool, project_of

from .trees import write_folder

DEF = '{"description": "Do it", "input_schema": {"type": "object"}}'
RUN = "def run(args, call):\n    return {}\n"
SERVER = "the server's own code; a tool imports the standard library, installed packages and modules beside it"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def declared(root: Path, *, body: str = RUN, definition: str = DEF, name: str = "fetch") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return write_folder(
        root,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "tools": {"include": [name]},
            }
        },
        tools={name: {"module": f"tools/{name}.py", "schema": f"tools/{name}.json"}},
        files={"a.md": "Work.\n", f"tools/{name}.json": definition, f"tools/{name}.py": body},
    )


def schema_of(tmp_path: Path, input_schema: dict[str, JsonValue]) -> str:
    return json.dumps({"description": "Do it", "input_schema": input_schema})


@pytest.mark.parametrize("key", ["input_schema", "output_schema"])
def test_a_schema_that_does_not_compile_is_refused_with_its_reason(
    tmp_path: Path, key: str
) -> None:
    bad = {"type": "object", "properties": {"n": {"minLength": -1}}}
    schemas = {"input_schema": {"type": "object"}, key: bad}
    folder_ = declared(tmp_path, definition=json.dumps({"description": "D", **schemas}))
    said = f"tools/folder: {folder_ / 'tools/fetch.json'} has an {key} that is not a valid JSON Schema: "
    with pytest.raises(
        ConfigurationError, match=rf"^{re.escape(said)}.*minimum.*; fix it in that file$"
    ):
        load_folder(folder_)


def test_a_schema_is_compiled_offline(tmp_path: Path) -> None:
    folder_ = declared(
        tmp_path,
        definition=schema_of(
            tmp_path, {"type": "object", "$ref": "https://example.invalid/o.json"}
        ),
    )
    with pytest.raises(ConfigurationError, match="Retrieval is disabled"):
        load_folder(folder_)


def test_a_2020_12_keyword_compiles(tmp_path: Path) -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"pair": {"type": "array", "prefixItems": [{"type": "string"}]}},
    }
    folder_ = declared(tmp_path, definition=schema_of(tmp_path, schema))
    assert load_folder(folder_)["a"].folder_tools["fetch"].input_schema == schema


@pytest.mark.parametrize("description", ["", "   ", None, 5])
def test_a_tool_without_a_description_in_text_is_refused_naming_the_file(
    tmp_path: Path, description: object
) -> None:
    definition: dict[str, object] = {"input_schema": {"type": "object"}}
    if description is not None:
        definition["description"] = description
    folder_ = declared(tmp_path, definition=json.dumps(definition))
    with pytest.raises(
        ConfigurationError,
        match=re.escape(f"{folder_ / 'tools/fetch.json'} gives no description"),
    ):
        load_folder(folder_)


@pytest.mark.parametrize(
    "schema",
    [{"type": "array"}, {"properties": {}}, {"type": ["object"]}],
)
def test_an_input_schema_whose_top_level_is_not_an_object_is_refused_naming_the_file(
    tmp_path: Path, schema: dict[str, JsonValue]
) -> None:
    folder_ = declared(tmp_path, definition=schema_of(tmp_path, schema))
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"{folder_ / 'tools/fetch.json'} has an input_schema whose top level is not"
        ),
    ):
        load_folder(folder_)


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
    folder_ = declared(tmp_path, definition=text)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(f"tools/folder: {folder_ / 'tools/fetch.json'} {said}"),
    ):
        load_folder(folder_)


def test_a_definition_or_a_script_that_is_not_utf8_is_refused_naming_it(tmp_path: Path) -> None:
    folder_ = declared(tmp_path)
    (folder_ / "tools/fetch.py").write_bytes(b"def run(args, call):\n    return '\xff'\n")
    with pytest.raises(
        ConfigurationError, match=re.escape(f"{folder_ / 'tools/fetch.py'} is not UTF-8")
    ):
        load_folder(folder_)


def test_a_script_importing_the_server_s_own_code_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path / "pyproject.toml", "[project]\nname = 'myapp'\n")
    write(tmp_path / "myapp/__init__.py", "SECRET = 1\n")
    agents = declared(tmp_path / "agents", body="import json\nfrom myapp import SECRET\n" + RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_folder(agents)
    assert str(caught.value) == (
        f"tools/folder: {agents / 'tools/fetch.py'} imports 'myapp' from {tmp_path / 'myapp/__init__.py'}, {SERVER}"
    )
    body = "import json\nimport urllib.request\nimport yaml\n" + RUN
    write(agents / "tools/fetch.py", body)
    assert list(load_folder(agents)["a"].folder_tools) == ["fetch"]


def test_a_folder_of_the_server_without_an_init_is_refused_naming_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path / "pyproject.toml", "")
    write(tmp_path / "pages/home.py", "TITLE = 'Home'\n")
    agents = declared(tmp_path / "agents", body="from pages.home import TITLE\n" + RUN)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(f"imports 'pages' from {tmp_path / 'pages'}, the server's own code"),
    ):
        load_folder(agents)


@pytest.mark.parametrize("name", ["tools", "questions"])
def test_a_server_package_named_like_a_declaration_folder_is_the_server_s_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.syspath_prepend(tmp_path / "src")
    write(tmp_path / "pyproject.toml", "[project]\nname = 'app'\n")
    write(tmp_path / f"src/{name}/__init__.py", "SECRET = 1\n")
    agents = declared(tmp_path / "agents", body=f"import {name}\n" + RUN)
    with pytest.raises(ConfigurationError) as caught:
        load_folder(agents)
    assert str(caught.value) == (
        f"tools/folder: {agents / 'tools/fetch.py'} imports {name!r} from {tmp_path / 'src' / name / '__init__.py'}, {SERVER}"
    )


def test_an_installed_package_inside_the_project_passes(tmp_path: Path) -> None:
    project = project_of(Path(yaml.__file__))
    assert Path(yaml.__file__).resolve().is_relative_to(project)
    write(tmp_path / "tools/fetch.py", "import yaml\n" + RUN)
    write(tmp_path / "tools/fetch.json", DEF)
    entry = ToolEntry.model_validate({"module": "tools/fetch.py", "schema": "tools/fetch.json"})
    assert load_tool(tmp_path, "fetch", entry, project).name == "fetch"


def test_a_site_packages_folder_of_the_project_that_is_not_the_interpreter_s_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages = tmp_path / "prefix/lib/python3.14/site-packages"
    monkeypatch.syspath_prepend(packages)
    write(tmp_path / "pyproject.toml", "")
    write(packages / "vendored/__init__.py", "")
    agents = declared(tmp_path / "agents", body="import vendored\n" + RUN)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"imports 'vendored' from {packages / 'vendored/__init__.py'}, the server's own code"
        ),
    ):
        load_folder(agents)


def test_a_site_folder_the_interpreter_adds_at_startup_inside_the_project_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    packages = tmp_path / "overlay/site-packages"
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    [own] = venv.glob("lib/python*/site-packages")
    write(own / "overlay.pth", f"import site; site.addsitedir({str(packages)!r})\n")
    monkeypatch.setattr(sys, "executable", str(venv / "bin/python"))
    monkeypatch.syspath_prepend(packages)
    folder._started.cache_clear()
    request.addfinalizer(folder._started.cache_clear)
    write(tmp_path / "pyproject.toml", "")
    write(packages / "vendored/__init__.py", "")
    agents = declared(tmp_path / "agents", body="import vendored\n" + RUN)
    assert list(load_folder(agents)["a"].folder_tools) == ["fetch"]


def test_a_folder_a_path_line_of_a_pth_adds_inside_the_project_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    source = tmp_path / "src"
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    [own] = venv.glob("lib/python*/site-packages")
    write(own / "editable.pth", f"{source}\n")
    monkeypatch.setattr(sys, "executable", str(venv / "bin/python"))
    monkeypatch.syspath_prepend(source)
    folder._started.cache_clear()
    request.addfinalizer(folder._started.cache_clear)
    write(tmp_path / "pyproject.toml", "")
    write(source / "server/__init__.py", "")
    agents = declared(tmp_path / "agents", body="import server\n" + RUN)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"imports 'server' from {source / 'server/__init__.py'}, the server's own code"
        ),
    ):
        load_folder(agents)


def _vendored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    packages = tmp_path / "vendor"
    monkeypatch.syspath_prepend(packages)
    write(tmp_path / "pyproject.toml", "")
    write(packages / "vendored/__init__.py", "")
    agents = declared(tmp_path / "agents", body="import vendored\n" + RUN)
    return packages, agents


def test_the_workers_interpreter_is_asked_only_when_the_server_s_site_folders_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    def refused(*args: object, **kwargs: object) -> None:
        raise AssertionError("the workers' interpreter was asked")

    packages, agents = _vendored(tmp_path, monkeypatch)
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(packages)])
    monkeypatch.setattr(subprocess, "run", refused)
    folder._installed.cache_clear()
    folder._started.cache_clear()
    request.addfinalizer(folder._installed.cache_clear)
    request.addfinalizer(folder._started.cache_clear)
    assert list(load_folder(agents)["a"].folder_tools) == ["fetch"]
    write(agents / "tools/fetch.py", "import json\n" + RUN)
    folder._installed.cache_clear()
    monkeypatch.setattr(site, "getsitepackages", lambda: [])
    assert list(load_folder(agents)["a"].folder_tools) == ["fetch"]


@pytest.mark.parametrize("failure", ["missing", "timeout", "noise"])
def test_a_workers_interpreter_that_cannot_answer_leaves_the_server_s_site_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, failure: str
) -> None:
    def late(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("python", 10)

    def noise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["python"], 0, stdout="hello from sitecustomize\n")

    packages, agents = _vendored(tmp_path, monkeypatch)
    if failure == "missing":
        monkeypatch.setattr(sys, "executable", str(tmp_path / "no-python"))
    else:
        monkeypatch.setattr(subprocess, "run", late if failure == "timeout" else noise)
    folder._started.cache_clear()
    request.addfinalizer(folder._started.cache_clear)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"imports 'vendored' from {packages / 'vendored/__init__.py'}, the server's own code"
        ),
    ):
        load_folder(agents)


def test_a_line_printed_before_the_workers_interpreter_answers_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    packages, agents = _vendored(tmp_path, monkeypatch)

    def answered(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        said = f"hello from sitecustomize\n{json.dumps([str(packages)])}\n"
        return subprocess.CompletedProcess(["python"], 0, stdout=said)

    monkeypatch.setattr(subprocess, "run", answered)
    folder._started.cache_clear()
    request.addfinalizer(folder._started.cache_clear)
    assert list(load_folder(agents)["a"].folder_tools) == ["fetch"]


def test_the_workers_interpreter_answers_past_a_startup_that_prints_late_or_lingers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    packages = tmp_path / "overlay/site-packages"
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    [own] = venv.glob("lib/python*/site-packages")
    write(
        own / "sitecustomize.py",
        "import atexit, pathlib, site, sys, threading, time\n"
        f"site.addsitedir(pathlib.Path({str(packages)!r}))\n"
        "sys.stdout.write('no newline')\n"
        "atexit.register(print, 0)\n"
        "threading.Thread(target=time.sleep, args=(60,)).start()\n",
    )
    monkeypatch.setattr(sys, "executable", str(venv / "bin/python"))
    folder._started.cache_clear()
    request.addfinalizer(folder._started.cache_clear)
    began = time.monotonic()
    assert packages.resolve() in folder._started()
    assert time.monotonic() - began < 5


def test_the_project_is_the_nearest_folder_with_a_pyproject_toml_else_the_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "agents/support").mkdir(parents=True)
    assert project_of(tmp_path / "agents/support") == (tmp_path / "agents/support").resolve()
    write(tmp_path / "pyproject.toml", "")
    assert project_of(tmp_path / "agents/support") == tmp_path.resolve()
    assert project_of(tmp_path) == tmp_path.resolve()


def test_a_module_imported_inside_a_function_is_checked_too(tmp_path: Path) -> None:
    body = "def run(args, call):\n    import json\n    import rfnry_engine\n    return {}\n"
    with pytest.raises(ConfigurationError, match="imports 'rfnry_engine'"):
        load_folder(declared(tmp_path, body=body))


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
    folder_ = declared(tmp_path, body=body)
    with pytest.raises(ConfigurationError) as caught:
        load_folder(folder_)
    assert str(caught.value) == (
        f"tools/folder: {folder_ / 'tools/fetch.py'} has no run at its top level; a tool defines def run(args, call) or async def run(args, call)"
    )


def test_a_script_that_is_not_valid_python_is_refused_naming_it(tmp_path: Path) -> None:
    folder_ = declared(tmp_path, body="def run(args, call:\n")
    with pytest.raises(
        ConfigurationError,
        match=re.escape(f"tools/folder: {folder_ / 'tools/fetch.py'} is not valid Python: "),
    ):
        load_folder(folder_)


@pytest.mark.parametrize(
    ("imports", "said"),
    [
        ("import rfnry_engine\n", "imports 'rfnry_engine'; a tool never imports the engine"),
        ("from myapp import SECRET\n", "imports 'myapp' from {server}, " + SERVER),
        (
            "import not_installed_anywhere\n",
            "imports 'not_installed_anywhere', which resolves to nothing; ",
        ),
        ("from . import other\n", "imports '.' relatively; import a module beside it"),
    ],
    ids=["the-engine", "the-server-s-code", "nothing", "relative"],
)
def test_a_module_beside_a_tool_keeps_the_rules_of_its_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imports: str, said: str
) -> None:
    monkeypatch.syspath_prepend(tmp_path)
    write(tmp_path / "pyproject.toml", "[project]\nname = 'myapp'\n")
    write(tmp_path / "myapp/__init__.py", "SECRET = 1\n")
    agents = declared(tmp_path / "agents", body="import shared\n" + RUN)
    write(agents / "tools/shared.py", imports + "VALUE = 1\n")
    with pytest.raises(ConfigurationError) as caught:
        load_folder(agents)
    expected = said.format(server=tmp_path / "myapp/__init__.py")
    assert str(caught.value).startswith(f"tools/folder: {agents / 'tools/shared.py'} {expected}")
