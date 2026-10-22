from collections.abc import Collection
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
        description = implementation.description if decl.description is None else decl.description
        if not description:
            raise ConfigurationError(
                f"declared: agent {spec.name!r} declares tool {name!r} with no description, in TOOLS.md or the Tool; describe it in either"
            )
        if decl.input is None:
            tools.append(
                Tool(
                    name,
                    description,
                    body=implementation.body,
                    args=implementation.args,
                    input_schema=implementation.input_schema,
                )
            )
            continue
        tools.append(
            Tool(
                name,
                description,
                body=implementation.body,
                args=_args_model(decl),
                input_schema=_schema(decl),
            )
        )
    return tools


def check_declarations(
    agents: dict[str, AgentSpec], code: dict[str, Tool], given: Collection[str]
) -> dict[str, list[Tool]]:
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
        if spec.model in given:
            built[key] = declared_tools(spec, code)
        declared.update(spec.tools)
    undeclared = sorted(name for name in code if name not in declared)
    if undeclared:
        raise ConfigurationError(
            f"declared: tools declared by no agent: {undeclared}; declare each in its agent's TOOLS.md or tools/"
        )
    return built
