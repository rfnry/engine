import ast
import importlib.util
import json
import site
import sys
import sysconfig
from collections.abc import Collection, Iterator
from functools import cache
from pathlib import Path

import jsonschema_rs
from pydantic import BaseModel, JsonValue

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.folders import declares, leads_out
from rfnry_engine.workers.pool import Script

ENGINE = "rfnry_engine"
KEYS = ("description", "input_schema", "output_schema")
NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class ToolDef(BaseModel):
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None
    script: Path
    folder: Path
    key: str
    source: str
    helpers: dict[str, str]
    awaits: bool = False


def script_key(root: Path, script: Path) -> str:
    return script.relative_to(root).with_suffix("").as_posix()


def project_of(root: Path) -> Path:
    place = Path(root).resolve()
    return next((p for p in [place, *place.parents] if (p / "pyproject.toml").is_file()), place)


@cache
def _installed() -> tuple[Path, ...]:
    paths = sysconfig.get_paths()
    places = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    return tuple(Path(place).resolve() for place in {*places, *site.getsitepackages()})


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(
            f"tools/folder: {path} is not UTF-8 text; save it as UTF-8"
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


def _definition(path: Path, script: Path, root: Path, sources: dict[Path, str]) -> ToolDef:
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
    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
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
    return ToolDef(
        name=path.stem,
        description=description,
        input_schema=arguments,
        output_schema=result,
        script=script,
        folder=path.parent,
        key=script_key(root, script),
        source=sources[script],
        helpers={p.stem: text for p, text in sources.items() if p.name.startswith("_")},
    )


def _imports(
    node: ast.AST, nested: bool = False
) -> Iterator[tuple[ast.Import | ast.ImportFrom, bool]]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Import | ast.ImportFrom):
            yield child, nested
        yield from _imports(child, nested or isinstance(child, NESTED))


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
    if any(place.is_relative_to(installed) for installed in _installed()):
        return False
    if not place.is_relative_to(project):
        return False
    return not any(place.is_relative_to(folder) for folder in folders)


def _parsed(path: Path, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ConfigurationError(
            f"tools/folder: {path} is not valid Python: {exc.msg} on line {exc.lineno}; fix it in that file"
        ) from exc


def _awaits(path: Path, module: ast.Module, what: str, given: str) -> bool:
    runs = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run"
    ]
    if not runs:
        raise ConfigurationError(
            f"tools/folder: {path} has no run at its top level; a {what} defines def run({given}, call) or async def run({given}, call)"
        )
    return isinstance(runs[-1], ast.AsyncFunctionDef)


def _beside(folder: Path) -> frozenset[str]:
    return frozenset(
        p.stem if p.is_file() else p.name
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
) -> None:
    held = f"{path.parent.name}/"
    for node, nested in _imports(module):
        if isinstance(node, ast.ImportFrom) and node.level:
            shown = "." * node.level + (node.module or "")
            raise ConfigurationError(
                f"tools/folder: {path} imports {shown!r} relatively; import a helper of its {held} folder by its name, such as import _client"
            )
        names = (
            [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else [a.name for a in node.names]
        )
        for name in names:
            top = name.partition(".")[0]
            if top == ENGINE:
                raise ConfigurationError(
                    f"tools/folder: {path} imports {ENGINE!r}; a {what} never imports the engine, so keep what it needs in the {what} or a helper beside it"
                )
            if top in helpers:
                if nested:
                    raise ConfigurationError(
                        f"tools/folder: {path} imports the helper {top!r} inside a function or class; import it at the top of the script, since a helper is reachable only while its folder's scripts load"
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
            for origin in origins:
                if _owned(origin, project, folders):
                    raise ConfigurationError(
                        f"tools/folder: {path} imports {top!r} from {origin}, the server's own code; a {what} imports the standard library, installed packages and helpers of its {held} folder"
                    )


def load_tools(
    folder: Path, project: Path, root: Path, folders: Collection[Path]
) -> dict[str, ToolDef]:
    folder, root, project = folder.absolute(), root.absolute(), project.resolve()
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    named = [p for p in files if not p.name.startswith("_")]
    helpers = [p for p in files if p.name.startswith("_") and p.suffix == ".py"]
    definitions = {p.stem: p for p in named if p.suffix == ".json"}
    scripts = {p.stem: p for p in named if p.suffix == ".py"}
    whole = "the agent's whole folder" if declares(folder.parent) else "the whole tools/ folder"
    for path in sorted([*definitions.values(), *scripts.values(), *helpers]):
        _held(path, folder, "its folder", whole)
    for name in sorted({*definitions, *scripts}):
        if name not in scripts:
            raise ConfigurationError(
                f"tools/folder: {definitions[name]} has no {name}.py beside it; a tool is {name}.json and {name}.py in a tools/ folder"
            )
        if name not in definitions:
            raise ConfigurationError(
                f"tools/folder: {scripts[name]} has no {name}.json beside it; describe the tool in {name}.json, or name a helper _{name}.py"
            )
    sources = {path: _text(path) for path in [*scripts.values(), *helpers]}
    tools = {
        name: _definition(definitions[name], scripts[name], root, sources)
        for name in sorted(scripts)
    }
    names = {p.stem for p in helpers}
    beside = _beside(folder)
    for path, source in sources.items():
        module = _parsed(path, source)
        _check_imports(path, module, project, folders, names, beside, "tool")
        if path in scripts.values():
            tools[path.stem].awaits = _awaits(path, module, "tool", "args")
    return tools


def load_scripts(
    folder: Path, names: Collection[str], project: Path, root: Path, folders: Collection[Path]
) -> dict[str, Script]:
    folder, root, project = folder.absolute(), root.absolute(), project.resolve()
    helpers = sorted(
        p for p in folder.iterdir() if p.is_file() and p.name.startswith("_") and p.suffix == ".py"
    )
    scripts = {name: folder / f"{name}.py" for name in names}
    for path in [*scripts.values(), *helpers]:
        _held(path, folder.parent, "the agent's folder", "the agent's whole folder")
    sources = {path: _text(path) for path in [*scripts.values(), *helpers]}
    shared = {p.stem: sources[p] for p in helpers}
    what = "question's script"
    awaits = {}
    beside = _beside(folder)
    for path, source in sources.items():
        module = _parsed(path, source)
        _check_imports(path, module, project, folders, shared, beside, what)
        if path in scripts.values():
            awaits[path] = _awaits(path, module, what, "input")
    return {
        name: Script(script_key(root, path), path, sources[path], shared, awaits[path])
        for name, path in scripts.items()
    }
