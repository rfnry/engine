import annotationlib
import ast
import dataclasses
import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

import rfnry_engine

API = Path(__file__).parents[4] / "docs" / "api.md"
TEXT = API.read_text(encoding="utf-8")
GROUPS = ["Models", "Engine", "Records", "Tools", "Storage", "Errors"]
HEADINGS: list[str] = re.findall(r"^### `(.+)`$", TEXT, re.M)
SECTIONS = re.findall(
    r"^### `([^`\n]+)`\n\n```python\n(.*?)```\n(.*?)(?=^##)", f"{TEXT}\n## ", re.M | re.S
)
BLOCKS = {heading: block for heading, block, _ in SECTIONS}
PROSE = {heading: prose for heading, _, prose in SECTIONS}
HOME = {
    alias.asname or alias.name: node.module
    for node in ast.parse(inspect.getsource(rfnry_engine)).body
    if isinstance(node, ast.ImportFrom) and node.module is not None
    for alias in node.names
}
EXPORTED = {name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__}
EMPTY = inspect.Parameter.empty
KINDS = inspect.Parameter


def is_protocol(value: object) -> bool:
    return inspect.isclass(value) and bool(getattr(value, "_is_protocol", False))


def verbs(cls: type) -> list[str]:
    return [n for n, v in vars(cls).items() if n[0] != "_" and inspect.isfunction(v)]


def members(protocol: type) -> set[str]:
    return {
        n
        for n, v in vars(protocol).items()
        if n[0] != "_" and (inspect.isfunction(v) or isinstance(v, property))
    }


PROTOCOLS = [value for value in EXPORTED.values() if is_protocol(value)]


def same(real: object, declared: object) -> bool:
    if isinstance(declared, property):
        return isinstance(real, property)
    return (
        inspect.isfunction(real)
        and callable(declared)
        and inspect.signature(real) == inspect.signature(declared)
    )


def implemented(cls: type) -> list[type]:
    return [
        p
        for p in PROTOCOLS
        if members(p) and all(n in vars(cls) and same(vars(cls)[n], vars(p)[n]) for n in members(p))
    ]


def is_class(value: object) -> bool:
    return (
        inspect.isclass(value)
        and not issubclass(value, BaseModel | BaseException)
        and not is_protocol(value)
    )


IMPLEMENTATIONS = {n: v for n, v in EXPORTED.items() if is_class(v) and implemented(v)}
CORE = {n: v for n, v in EXPORTED.items() if is_class(v) and not implemented(v)}
VERBS = [f"{name}.{verb}" for name, cls in CORE.items() for verb in verbs(cls)]


def defined_at_top(module: ModuleType) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(ast.unparse(target) for target in node.targets)
    return names


MODULES = [
    importlib.import_module(info.name)
    for info in pkgutil.walk_packages(rfnry_engine.__path__, "rfnry_engine.")
]
PRIVATE = {
    name: module for module in MODULES for name in defined_at_top(module) if name not in EXPORTED
}


SOURCE = annotationlib.Format.STRING


def normal(annotation: object) -> str:
    if annotation is EMPTY:
        return ""
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return ast.unparse(ast.parse(text, mode="eval"))


def written(node: ast.expr | None) -> str:
    return "" if node is None else ast.unparse(node)


def value(node: ast.expr | None) -> Any:
    return EMPTY if node is None else ast.literal_eval(node)


Function = ast.FunctionDef | ast.AsyncFunctionDef


def documented(node: Function) -> list[tuple[str, Any, str, Any]]:
    a = node.args
    positional = [*a.posonlyargs, *a.args]
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    rows: list[tuple[str, Any, str, Any]] = [
        (
            arg.arg,
            KINDS.POSITIONAL_ONLY if arg in a.posonlyargs else KINDS.POSITIONAL_OR_KEYWORD,
            written(arg.annotation),
            value(default),
        )
        for arg, default in zip(positional, defaults, strict=True)
    ]
    if a.vararg is not None:
        rows.append((a.vararg.arg, KINDS.VAR_POSITIONAL, written(a.vararg.annotation), EMPTY))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        rows.append((arg.arg, KINDS.KEYWORD_ONLY, written(arg.annotation), value(default)))
    if a.kwarg is not None:
        rows.append((a.kwarg.arg, KINDS.VAR_KEYWORD, written(a.kwarg.annotation), EMPTY))
    return rows


def check_function(node: Function, real: Any) -> None:
    signature = inspect.signature(real, annotation_format=SOURCE)
    expected = [
        (p.name, p.kind, normal(p.annotation), p.default) for p in signature.parameters.values()
    ]
    assert documented(node) == expected, node.name
    assert written(node.returns) == normal(signature.return_annotation), node.name
    assert isinstance(node, ast.AsyncFunctionDef) == inspect.iscoroutinefunction(real), node.name


def fields(cls: Any) -> dict[str, tuple[str, Any]]:
    declared = {}
    for klass in reversed(cls.__mro__):
        declared.update(inspect.get_annotations(klass, format=SOURCE))
    if issubclass(cls, BaseModel):
        return {
            name: (
                normal(declared[name]),
                EMPTY if f.is_required() else f.get_default(call_default_factory=True),
            )
            for name, f in cls.model_fields.items()
        }
    return {
        f.name: (
            normal(declared[f.name]),
            f.default
            if f.default is not dataclasses.MISSING
            else EMPTY
            if f.default_factory is dataclasses.MISSING
            else f.default_factory(),
        )
        for f in dataclasses.fields(cls)
    }


def check_class(node: ast.ClassDef, cls: Any) -> None:
    ancestors = {klass.__name__ for klass in cls.__mro__}
    bases = [ast.unparse(base) for base in node.bases]
    assert all(base in ancestors and base[0] != "_" for base in bases), node.name
    attributes = {ast.unparse(s.target): s for s in node.body if isinstance(s, ast.AnnAssign)}
    methods = {s.name: s for s in node.body if isinstance(s, Function)}
    for name, method in methods.items():
        check_function(method, getattr(cls, name))
    properties: dict[str, Any] = {
        n: v for n, v in vars(cls).items() if n[0] != "_" and isinstance(v, property)
    }
    if issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls):
        real = fields(cls)
        assert [name for name in attributes if name not in properties] == list(real), node.name
        for name, s in attributes.items():
            if name in properties:
                returned = inspect.get_annotations(properties[name].fget, format=SOURCE)["return"]
                assert written(s.annotation) == normal(returned), f"{node.name}.{name}"
            else:
                assert (written(s.annotation), value(s.value)) == real[name], f"{node.name}.{name}"
        return
    if getattr(cls, "_is_protocol", False):
        called = {"__call__"} & set(vars(cls))
        assert set(attributes) | set(methods) == set(properties) | set(verbs(cls)) | called, (
            node.name
        )
    assert set(properties) <= set(attributes), node.name
    for name, s in attributes.items():
        if name in properties:
            returned = inspect.get_annotations(properties[name].fget, format=SOURCE)["return"]
            assert written(s.annotation) == normal(returned), f"{node.name}.{name}"
        else:
            assert f"self.{name} = " in inspect.getsource(cls), f"{node.name}.{name}"


def aliases(module: ModuleType) -> dict[str, str]:
    return {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.parse(inspect.getsource(module)).body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }


def defined(node: ast.stmt) -> str:
    if isinstance(node, ast.Assign):
        return ast.unparse(node.targets[0])
    assert isinstance(node, ast.ClassDef | Function), ast.unparse(node)
    return node.name


def test_the_groups_come_in_order() -> None:
    groups = re.findall(r"^## (.+)$", TEXT, re.M)
    assert groups == [group for group in GROUPS if group in groups]


def test_every_public_name_has_one_entry_and_every_entry_names_something_public() -> None:
    assert len(HEADINGS) == len(set(HEADINGS))
    assert sorted(set(rfnry_engine.__all__) - set(HEADINGS)) == []
    assert sorted(set(HEADINGS) - set(rfnry_engine.__all__) - set(VERBS)) == []


def test_every_verb_of_a_core_class_has_an_entry_right_after_its_class() -> None:
    assert sorted(set(VERBS) - set(HEADINGS)) == []
    for before, heading in zip(HEADINGS, HEADINGS[1:], strict=False):
        owner, dot, _ = heading.partition(".")
        if dot:
            assert before == owner or before.startswith(f"{owner}."), heading


@pytest.mark.parametrize("heading", HEADINGS)
def test_the_entry_matches_the_code(heading: str) -> None:
    body = ast.parse(BLOCKS[heading]).body
    owner, _, verb = heading.partition(".")
    if verb:
        [node] = body
        assert isinstance(node, Function) and node.name == verb, heading
        check_function(node, getattr(getattr(rfnry_engine, owner), verb))
        return
    assert owner in [defined(node) for node in body], heading
    module = importlib.import_module(HOME[owner])
    for node in body:
        if isinstance(node, ast.Assign):
            assert ast.unparse(node.value) == aliases(module).get(defined(node)), heading
        elif isinstance(node, ast.ClassDef):
            check_class(node, getattr(module, node.name))
        else:
            assert isinstance(node, Function), heading
            check_function(node, getattr(module, node.name))


def test_the_storage_implements_exactly_its_protocol() -> None:
    assert rfnry_engine.Storage in implemented(rfnry_engine.PostgresStorage)
    assert rfnry_engine.Backend in implemented(rfnry_engine.Local)
    for name, cls in IMPLEMENTATIONS.items():
        promised = set[str]().union(*(members(protocol) for protocol in implemented(cls)))
        assert sorted(verbs(cls)) == sorted(promised), name
        for protocol in implemented(cls):
            for verb in members(protocol):
                real, declared = getattr(cls, verb), getattr(protocol, verb)
                assert inspect.signature(real) == inspect.signature(declared), f"{name}.{verb}"
                assert inspect.iscoroutinefunction(real) == inspect.iscoroutinefunction(declared)


def used(block: str) -> set[str]:
    tree = ast.parse(block)
    names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return names - {defined(node) for node in tree.body}


@pytest.mark.parametrize("heading", HEADINGS)
def test_a_private_type_in_a_signature_is_spelled_out_in_the_prose(heading: str) -> None:
    for name in sorted(used(BLOCKS[heading]) & set(PRIVATE)):
        spelled = re.search(rf"`{name}` is (?:`([^`]+)`)?", PROSE[heading])
        assert spelled is not None, f"{heading}: {name}"
        value = aliases(PRIVATE[name]).get(name)
        if value is not None:
            assert spelled[1] is not None and normal(spelled[1]) == value, f"{heading}: {name}"
