import ast
import dataclasses
import importlib
import inspect
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
BLOCKS = dict(re.findall(r"^### `([^`\n]+)`\n\n```python\n(.*?)```", TEXT, re.M | re.S))
HOME = {
    alias.asname or alias.name: node.module
    for node in ast.parse(inspect.getsource(rfnry_engine)).body
    if isinstance(node, ast.ImportFrom) and node.module is not None
    for alias in node.names
}
EMPTY = inspect.Parameter.empty
KINDS = inspect.Parameter


def is_core(value: object) -> bool:
    return (
        inspect.isclass(value)
        and not issubclass(value, BaseModel | BaseException)
        and not getattr(value, "_is_protocol", False)
    )


CORE = {
    n: getattr(rfnry_engine, n) for n in rfnry_engine.__all__ if is_core(getattr(rfnry_engine, n))
}


def verbs(cls: type) -> list[str]:
    return [n for n, v in vars(cls).items() if n[0] != "_" and inspect.isfunction(v)]


VERBS = [f"{name}.{verb}" for name, cls in CORE.items() for verb in verbs(cls)]


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
    signature = inspect.signature(real)
    expected = [
        (p.name, p.kind, normal(p.annotation), p.default) for p in signature.parameters.values()
    ]
    assert documented(node) == expected, node.name
    assert written(node.returns) == normal(signature.return_annotation), node.name
    assert isinstance(node, ast.AsyncFunctionDef) == inspect.iscoroutinefunction(real), node.name


def fields(cls: Any) -> dict[str, tuple[str, Any]]:
    declared = {}
    for klass in reversed(cls.__mro__):
        declared.update(inspect.get_annotations(klass))
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
            normal(f.type),
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
    assert all(ast.unparse(base) in ancestors for base in node.bases), node.name
    attributes = {ast.unparse(s.target): s for s in node.body if isinstance(s, ast.AnnAssign)}
    methods = {s.name: s for s in node.body if isinstance(s, Function)}
    for name, method in methods.items():
        check_function(method, getattr(cls, name))
    if issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls):
        real = fields(cls)
        assert list(attributes) == list(real), node.name
        for name, s in attributes.items():
            assert (written(s.annotation), value(s.value)) == real[name], f"{node.name}.{name}"
        return
    properties: dict[str, Any] = {
        n: v for n, v in vars(cls).items() if n[0] != "_" and isinstance(v, property)
    }
    if getattr(cls, "_is_protocol", False):
        assert set(attributes) | set(methods) == set(properties) | set(verbs(cls)), node.name
    assert set(properties) <= set(attributes), node.name
    for name, s in attributes.items():
        if name in properties:
            returned = inspect.get_annotations(properties[name].fget)["return"]
            assert written(s.annotation) == normal(returned), f"{node.name}.{name}"
        else:
            assert f"self.{name} = " in inspect.getsource(cls), f"{node.name}.{name}"


def alias(module: ModuleType, name: str) -> str:
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == [name]:
            return ast.unparse(node.value)
    raise AssertionError(f"{name} is not an alias in {module.__name__}")


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
            assert ast.unparse(node.value) == alias(module, defined(node)), heading
        elif isinstance(node, ast.ClassDef):
            check_class(node, getattr(module, node.name))
        else:
            assert isinstance(node, Function), heading
            check_function(node, getattr(module, node.name))
