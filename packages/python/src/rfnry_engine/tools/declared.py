import json
from collections.abc import Collection
from functools import cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.tools.builtins import BUILTINS
from rfnry_engine.tools.delegate import DELEGATE_TOOL
from rfnry_engine.tools.history import HISTORY_TOOL
from rfnry_engine.tools.tools import Tool
from rfnry_engine.tools.workspace import WORKSPACE_NAMES
from rfnry_engine.tree import AgentSpec, ToolDecl

_PYTHON: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
}

_DATA = {"default", "examples", "const", "enum"}
_NAMED = {"properties", "$defs", "definitions", "patternProperties"}


def prose(schema: object, at: str = "") -> str | None:
    return _prose(schema, at, False)


def _prose(schema: object, at: str, names: bool) -> str | None:
    if isinstance(schema, dict):
        for key, value in schema.items():
            if not names and key in _DATA:
                continue
            here = f"{at}.{key}" if at else str(key)
            if not names and key == "description" and isinstance(value, str):
                return here
            found = _prose(value, here, not names and key in _NAMED)
            if found is not None:
                return found
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            found = _prose(item, f"{at}.{index}" if at else str(index), False)
            if found is not None:
                return found
    return None


@cache
def described(args: type[BaseModel]) -> str:
    return json.dumps(args.model_json_schema())


def _schema(decl: ToolDecl) -> dict[str, Any]:
    fields = decl.input or {}
    return {
        "type": "object",
        "properties": {
            name: {"type": f.type, "description": f.description} for name, f in fields.items()
        },
        "required": [name for name, f in fields.items() if f.required],
    }


def _args_model(decl: ToolDecl) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for at, (name, field) in enumerate((decl.input or {}).items()):
        python = _PYTHON[field.type]
        fields[f"field_{at}"] = (
            (python, Field(alias=name))
            if field.required
            else (python | None, Field(default=None, alias=name))
        )
    config = ConfigDict(
        extra="forbid", serialize_by_alias=True, validate_by_alias=True, validate_by_name=False
    )
    return create_model(f"{decl.name}_args", __config__=config, **fields)


def declared_tools(spec: AgentSpec, code: dict[str, Tool]) -> list[Tool]:
    tools: list[Tool] = []
    for name, decl in spec.tools.items():
        implementation = code.get(name)
        if implementation is None or implementation.body is None:
            raise ConfigurationError(
                f"declared: agent {spec.name!r} declares tool {name!r} but no Tool with a body was given; pass Engine(tools=) a Tool named {name!r} with a body"
            )
        if decl.input is None:
            tools.append(
                Tool(
                    name,
                    decl.description,
                    body=implementation.body,
                    args=implementation.args,
                    input_schema=implementation.input_schema,
                )
            )
            continue
        tools.append(
            Tool(
                name,
                decl.description,
                body=implementation.call
                if implementation.args is not None
                else implementation.body,
                args=_args_model(decl),
                input_schema=_schema(decl),
            )
        )
    return tools


def _check_code(code: dict[str, Tool]) -> None:
    for name, tool in code.items():
        if tool.description:
            raise ConfigurationError(
                f"declared: tool {name!r} carries a description in code; describe it in its agent's TOOLS.md or tools/{name}.md"
            )
        schema = tool.input_schema
        if schema is None and tool.args is not None:
            schema = json.loads(described(tool.args))
        where = prose(schema)
        if where is not None:
            raise ConfigurationError(
                f"declared: tool {name!r} carries a description in code at {where}; describe its arguments under input: in tools/{name}.md"
            )


def _check_input(spec: AgentSpec, name: str, decl: ToolDecl, tool: Tool) -> None:
    if decl.input is None:
        return
    if tool.input_schema is not None:
        raise ConfigurationError(
            f"declared: agent {spec.name!r} gives tool {name!r} an input: in tools/{name}.md and its Tool an input_schema; keep one, or use args= beside input:"
        )
    if tool.args is not None:
        written = sorted(decl.input)
        coded = sorted(json.loads(described(tool.args)).get("properties", {}))
        if written != coded:
            raise ConfigurationError(
                f"declared: agent {spec.name!r} gives tool {name!r} the arguments {written} in tools/{name}.md and {coded} in its args; make them name the same arguments"
            )


def check_declarations(
    agents: dict[str, AgentSpec], code: dict[str, Tool], given: Collection[str]
) -> dict[str, list[Tool]]:
    _check_code(code)
    declared: set[str] = set()
    built: dict[str, list[Tool]] = {}
    for key, spec in agents.items():
        taken = {
            *BUILTINS,
            HISTORY_TOOL,
            *([DELEGATE_TOOL] if spec.agents else []),
            *(WORKSPACE_NAMES if spec.workspace else []),
        }
        clash = sorted(name for name in spec.tools if name in taken)
        if clash:
            raise ConfigurationError(
                f"declared: agent {spec.name!r} declares {', '.join(map(repr, clash))}, the name of a tool the engine gives it; rename yours"
            )
        for name, decl in spec.tools.items():
            if name in code:
                _check_input(spec, name, decl, code[name])
        if spec.model in given:
            built[key] = declared_tools(spec, code)
        declared.update(spec.tools)
    undeclared = sorted(name for name in code if name not in declared)
    if undeclared:
        raise ConfigurationError(
            f"declared: tools declared by no agent: {undeclared}; declare each in its agent's TOOLS.md or tools/"
        )
    return built
