from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.tools.tools import Tool
from rfnry_engine.tree import AgentSpec, ToolDecl

_PYTHON: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
}


def _schema(decl: ToolDecl) -> dict[str, Any]:
    fields = decl.input or {}
    return {
        "type": "object",
        "properties": {name: {"type": f.type, "description": f.description} for name, f in fields.items()},
        "required": [name for name, f in fields.items() if f.required],
    }


def _args_model(decl: ToolDecl) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, field in (decl.input or {}).items():
        python = _PYTHON[field.type]
        fields[name] = (python, ...) if field.required else (python | None, None)
    return create_model(f"{decl.name}_args", __config__=ConfigDict(extra="forbid"), **fields)


def declared_tools(spec: AgentSpec, code: dict[str, Tool]) -> list[Tool]:
    tools: list[Tool] = []
    for name, decl in spec.tools.items():
        implementation = code.get(name)
        if implementation is None or implementation.body is None:
            raise ConfigurationError(f"declared: agent {spec.name!r} declares tool {name!r} but no Tool with a body was given")
        if decl.description is None:
            if not implementation.description:
                raise ConfigurationError(f"declared: agent {spec.name!r} adopts tool {name!r} by name, but the Tool has no description")
            tools.append(implementation)
            continue
        tools.append(Tool(name, decl.description, body=implementation.body, args=_args_model(decl), input_schema=_schema(decl)))
    return tools


def check_declarations(agents: dict[str, AgentSpec], code: dict[str, Tool], *, bodies: bool) -> None:
    declared: set[str] = set()
    for spec in agents.values():
        if bodies:
            declared_tools(spec, code)
        declared.update(spec.tools)
    undeclared = sorted(name for name in code if name not in declared)
    if undeclared:
        raise ConfigurationError(f"declared: tools declared by no agent: {undeclared}")
