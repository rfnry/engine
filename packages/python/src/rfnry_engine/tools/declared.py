import json
from collections.abc import Collection
from functools import cache

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.tools.builtins import BUILTINS
from rfnry_engine.tools.delegate import DELEGATE_TOOL
from rfnry_engine.tools.history import HISTORY_TOOL
from rfnry_engine.tools.tools import Tool
from rfnry_engine.tools.workspace import WORKSPACE_NAMES
from rfnry_engine.tree import AgentSpec

_DATA = {"default", "examples", "const", "enum", "discriminator"}
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


def declared_tools(spec: AgentSpec, code: dict[str, Tool]) -> list[Tool]:
    tools: list[Tool] = []
    for name in sorted(spec.tools):
        tool = code.get(name)
        if tool is None:
            raise ConfigurationError(
                f"declared: agent {spec.name!r} lists tool {name!r} but no Tool was given; pass Engine(tools=) a Tool named {name!r}"
            )
        tools.append(tool)
    return tools


def check_declarations(
    agents: dict[str, AgentSpec], code: dict[str, Tool], given: Collection[str]
) -> dict[str, list[Tool]]:
    for name, tool in code.items():
        if tool.body is None:
            raise ConfigurationError(
                f"declared: tool {name!r} has no body; give its Tool the function that runs it"
            )
        if not tool.description.strip():
            raise ConfigurationError(
                f"declared: tool {name!r} has no description; give its Tool one, since nothing else describes it to the model"
            )
    listed: set[str] = set()
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
                f"declared: agent {spec.name!r} lists {clash[0]!r}, a tool the engine gives it; drop it from tools:"
            )
        if spec.model in given:
            built[key] = declared_tools(spec, code)
        listed.update(spec.tools)
    unlisted = sorted(name for name in code if name not in listed)
    if unlisted:
        raise ConfigurationError(
            f"declared: tools no agent lists: {unlisted}; list each in the tools: of the agent that uses it"
        )
    return built
