import ast
import importlib.util
import json
import site
import subprocess
import sys
import sysconfig
from collections.abc import Collection, Iterator
from functools import cache
from importlib.machinery import FrozenImporter
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import jsonschema_rs
from pydantic import BaseModel, JsonValue

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.folders import leads_out
from rfnry_engine.manifest import ToolEntry
from rfnry_engine.workers.pool import Helper, Script

ENGINE = "rfnry_engine"
KEYS = ("description", "input_schema", "output_schema")
NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError"})


class ToolDef(BaseModel):
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None
    script: Path
    folder: Path
    key: str
    source: str
    helpers: dict[str, Helper]
    function: str = "run"
    timeout: float | None = None


class _Schema(NamedTuple):
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None


def script_key(root: Path, script: Path) -> str:
    return script.relative_to(root).with_suffix("").as_posix()


def project_of(root: Path) -> Path:
    place = Path(root).resolve()
    return next((p for p in [place, *place.parents] if (p / "pyproject.toml").is_file()), place)


SITE = """
import json, os, site, sys
seen = []
add = site.addsitedir
def record(sitedir, known_paths=None):
    seen.append(os.fsdecode(sitedir))
    return add(sitedir, known_paths)
site.addsitedir = record
site.main()
sys.stdout.write("\\n" + json.dumps(seen) + "\\n")
sys.stdout.flush()
os._exit(0)
"""
PROBE = 10.0


@cache
def _installed() -> tuple[Path, ...]:
    paths = sysconfig.get_paths()
    places = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    return tuple(Path(place).resolve() for place in {*places, *site.getsitepackages()})


@cache
def _started() -> tuple[Path, ...]:
    try:
        found = subprocess.run(
            [sys.executable, "-P", "-S", "-c", SITE],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PROBE,
            check=False,
        )
        return tuple(Path(place).resolve() for place in json.loads(found.stdout.splitlines()[-1]))
    except OSError, subprocess.SubprocessError, ValueError, IndexError, TypeError:
        return ()


def _among(place: Path, places: tuple[Path, ...]) -> bool:
    return any(place.is_relative_to(held) for held in places)


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(
            f"tools/folder: {path} is not UTF-8 text; save it as UTF-8"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"tools/folder: cannot read {path}: {exc.strerror or exc}; make it readable to the engine"
        ) from exc


def reason(exc: jsonschema_rs.ValidationError) -> str:
    at = "/".join(str(step) for step in exc.instance_path)
    return f"{exc.message} at {at}" if at else exc.message


def _compiled(path: Path, key: str, schema: dict[str, JsonValue]) -> None:
    try:
        jsonschema_rs.validator_for(schema, offline=True)
    except jsonschema_rs.ValidationError as exc:
        raise ConfigurationError(
            f"tools/folder: {path} has an {key} that is not a valid JSON Schema: {reason(exc)}; fix it in that file"
        ) from exc


def _schema_file(path: Path, given: str | None) -> _Schema:
    try:
        data = json.loads(_text(path))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"tools/folder: {path} is not valid JSON: {exc}; fix it in that file"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"tools/folder: {path} is not a JSON object; a tool's JSON is an object that takes description, input_schema and output_schema"
        )
    unknown = sorted(key for key in data if key not in KEYS)
    if unknown:
        raise ConfigurationError(
            f"tools/folder: {path} has the unknown key {unknown[0]!r}; a tool's JSON takes description, input_schema and output_schema"
        )
    said: Any = data.get("description")
    if (given is None or "description" in data) and (not isinstance(said, str) or not said.strip()):
        raise ConfigurationError(
            f"tools/folder: {path} gives no description in text; write what the tool does and what it returns, since nothing else describes it to the model"
        )
    arguments = data.get("input_schema")
    if not isinstance(arguments, dict):
        raise ConfigurationError(
            f'tools/folder: {path} has no input_schema object; give the tool\'s arguments as a JSON Schema with "type": "object"'
        )
    _compiled(path, "input_schema", arguments)
    if arguments.get("type") != "object":
        raise ConfigurationError(
            f'tools/folder: {path} has an input_schema whose top level is not "type": "object"; a tool takes its arguments as an object, so give its input_schema "type": "object"'
        )
    result = data.get("output_schema")
    if result is not None and not isinstance(result, dict):
        raise ConfigurationError(
            f"tools/folder: {path} has an output_schema that is not an object; write it as a JSON Schema object, or leave it out"
        )
    if result is not None:
        _compiled(path, "output_schema", result)
    return _Schema(given or said, arguments, result)


def _catches_import_errors(handler: ast.ExceptHandler) -> bool:
    caught = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(kind, ast.Name) and kind.id in IMPORT_ERRORS for kind in caught)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _guarded_body(node: ast.AST) -> list[ast.stmt]:
    if isinstance(node, ast.Try | ast.TryStar) and any(map(_catches_import_errors, node.handlers)):
        return node.body
    if isinstance(node, ast.If) and _is_type_checking(node.test):
        return node.body
    return []


def _imports(
    node: ast.AST, nested: bool = False, guarded: bool = False
) -> Iterator[tuple[ast.Import | ast.ImportFrom, bool, bool]]:
    safe = _guarded_body(node)
    for child in ast.iter_child_nodes(node):
        kept = guarded or child in safe
        if isinstance(child, ast.Import | ast.ImportFrom):
            yield child, nested, kept
        yield from _imports(child, nested or isinstance(child, NESTED), kept)


def _imported(node: ast.Import | ast.ImportFrom) -> list[str]:
    return [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]


def _origins(name: str) -> list[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except ImportError, ValueError:
        return []
    if spec is None or spec.origin in {"built-in", "frozen"}:
        return []
    if spec.has_location and spec.origin is not None:
        return [Path(spec.origin)]
    return [Path(place) for place in spec.submodule_search_locations or []]


def _owned(origin: Path, project: Path, folders: Collection[Path]) -> bool:
    place = origin.resolve()
    if not place.is_relative_to(project):
        return False
    if any(place.is_relative_to(folder) for folder in folders):
        return False
    return not (_among(place, _installed()) or _among(place, _started()))


def _parsed(path: Path, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ConfigurationError(
            f"tools/folder: {path} is not valid Python: {exc.msg} on line {exc.lineno}; fix it in that file"
        ) from exc


def _check_run(
    path: Path, module: ast.Module, what: str, given: str, function: str = "run"
) -> None:
    if not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function
        for node in module.body
    ):
        raise ConfigurationError(
            f"tools/folder: {path} has no {function} at its top level; a {what} defines def {function}({given}, call) or async def {function}({given}, call)"
        )


def _standard(name: str) -> bool:
    return (
        name in sys.stdlib_module_names
        or name in sys.builtin_module_names
        or FrozenImporter.find_spec(name) is not None
    )


def _check_name(path: Path) -> None:
    name = path.stem
    if _standard(name):
        raise ConfigurationError(
            f"tools/folder: {path} is named like the standard-library module {name!r}, and the worker can't load both; rename it"
        )


def _beside(folder: Path) -> frozenset[str]:
    return frozenset(
        p.stem if p.suffix == ".py" and not p.is_dir() else p.name
        for p in folder.iterdir()
        if not p.name.startswith((".", "_")) and (p.is_dir() or p.suffix == ".py")
    )


def _held(path: Path, folder: Path, named: str, whole: str) -> None:
    target = leads_out(path, folder)
    if target is not None:
        raise ConfigurationError(
            f"tools/folder: {path} links to {target}, outside {named}; move the file into {named}, or link {whole}"
        )


def _check_imports(
    path: Path,
    module: ast.Module,
    project: Path,
    folders: Collection[Path],
    helpers: Collection[str],
    beside: Collection[str],
    what: str,
    declared: bool = False,
) -> None:
    held = f"{path.parent.name}/"
    importable = "modules beside it" if declared else f"helpers of its {held} folder"
    example = (
        "a module beside it by its name, such as import shared"
        if declared
        else f"a helper of its {held} folder by its name, such as import _client"
    )
    for node, nested, guarded in _imports(module):
        if isinstance(node, ast.ImportFrom) and node.level:
            shown = "." * node.level + (node.module or "")
            raise ConfigurationError(
                f"tools/folder: {path} imports {shown!r} relatively; import {example}"
            )
        for name in _imported(node):
            top = name.partition(".")[0]
            if top == ENGINE:
                raise ConfigurationError(
                    f"tools/folder: {path} imports {ENGINE!r}; a {what} never imports the engine, so keep what it needs in the {what} or a helper beside it"
                )
            if declared and (path.parent / top / "__init__.py").is_file():
                raise ConfigurationError(
                    f"tools/folder: {path} imports {top!r}, a package folder beside it; a module beside a tool is a .py file there, so make it {top}.py, or install the package"
                )
            if top in helpers:
                if nested:
                    raise ConfigurationError(
                        f"tools/folder: {path} imports the helper {top!r} inside a function or class; import it at the top of the script, since a helper is reachable only while its folder's scripts load"
                    )
                if declared and "." in name and not guarded:
                    raise ConfigurationError(
                        f"tools/folder: {path} imports {name!r}, and {top!r} beside it is a .py file with nothing below it; import {top} and use what it holds"
                    )
                continue
            origins = _origins(top)
            if (
                top in beside
                and top not in sys.stdlib_module_names
                and all(path.parent.resolve() in o.resolve().parents for o in origins)
            ):
                raise ConfigurationError(
                    f"tools/folder: {path} imports {top!r}, which sits in its {held} folder but is not a helper; a {what} reaches only the helpers of its folder, so move the shared code into a helper _{top}.py"
                )
            if declared and not origins and not guarded and not _standard(top):
                raise ConfigurationError(
                    f"tools/folder: {path} imports {top!r}, which resolves to nothing; a module beside a {what} is a .py file there, and otherwise it reaches the standard library and installed packages, so install it, move it there as a .py file, or guard the import with try/except ImportError or if TYPE_CHECKING"
                )
            for origin in origins:
                if _owned(origin, project, folders):
                    raise ConfigurationError(
                        f"tools/folder: {path} imports {top!r} from {origin}, the server's own code; a {what} imports the standard library, installed packages and {importable}"
                    )


def _file_of(root: Path, name: str, written: str, what: str) -> Path:
    path = root / written
    if not path.is_file():
        raise ConfigurationError(
            f"tools/folder: tool {name!r} names the {what} {written!r}, and the folder holds no such file; write its path from the folder root"
        )
    _held(path, root, "the folder root", "the whole folder")
    return path


def _with_its_modules(
    script: Path, source: str, root: Path
) -> tuple[dict[Path, ast.Module], dict[str, Helper]]:
    parsed = {script: _parsed(script, source)}
    helpers: dict[str, Helper] = {}
    waiting = [parsed[script]]
    while waiting:
        for node, _, _ in _imports(waiting.pop()):
            for name in _imported(node):
                top = name.partition(".")[0]
                place = script.parent / f"{top}.py"
                if top in helpers or not place.is_file():
                    continue
                _held(place, root, "the folder root", "the whole folder")
                _check_name(place)
                helpers[top] = Helper(place, _text(place))
                parsed[place] = _parsed(place, helpers[top].source)
                waiting.append(parsed[place])
    return parsed, helpers


def _schema_of(root: Path, name: str, entry: ToolEntry) -> _Schema:
    if entry.schema_file is not None:
        return _schema_file(_file_of(root, name, entry.schema_file, "schema"), entry.description)
    if entry.description is None:
        raise ConfigurationError(
            f"tools/folder: tool {name!r} has no description; write description: on its entry or give it a schema file that has one, since nothing else describes it to the model"
        )
    return _Schema(entry.description, {"type": "object"}, None)


def load_tool(root: Path, name: str, entry: ToolEntry, project: Path) -> ToolDef:
    root, project = root.absolute(), project.resolve()
    if PurePosixPath(entry.module).suffix != ".py":
        raise ConfigurationError(
            f"tools/folder: tool {name!r} has the module {entry.module!r}, which is not a .py file; the workers run Python, so write the path of a .py file from the folder root"
        )
    script = _file_of(root, name, entry.module, "module")
    source = _text(script)
    parsed, helpers = _with_its_modules(script, source, root)
    for path, module in parsed.items():
        _check_imports(path, module, project, (), helpers, (), "tool", declared=True)
    _check_run(script, parsed[script], "tool", "args", entry.function)
    schema = _schema_of(root, name, entry)
    return ToolDef(
        name=name,
        description=schema.description,
        input_schema=schema.input_schema,
        output_schema=schema.output_schema,
        script=script,
        folder=script.parent,
        key=script_key(root, script),
        source=source,
        helpers=helpers,
        function=entry.function,
        timeout=entry.timeout,
    )


def load_scripts(
    folder: Path, names: Collection[str], project: Path, root: Path, folders: Collection[Path]
) -> dict[str, Script]:
    folder, root, project = folder.absolute(), root.absolute(), project.resolve()
    helpers = sorted(
        p for p in folder.iterdir() if p.is_file() and p.name.startswith("_") and p.suffix == ".py"
    )
    scripts = {name: folder / f"{name}.py" for name in names}
    for path in [*scripts.values(), *helpers]:
        _held(path, root, "the folder root", "the whole folder")
    for path in helpers:
        _check_name(path)
    sources = {path: _text(path) for path in [*scripts.values(), *helpers]}
    shared = {p.stem: Helper(p, sources[p]) for p in helpers}
    what = "question's script"
    beside = _beside(folder)
    for path, source in sources.items():
        module = _parsed(path, source)
        _check_imports(path, module, project, folders, shared, beside, what)
        if path in scripts.values():
            _check_run(path, module, what, "input")
    return {
        name: Script(script_key(root, path), path, sources[path], shared)
        for name, path in scripts.items()
    }
