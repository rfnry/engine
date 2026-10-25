import json
from copy import deepcopy
from functools import cache
from typing import Any, cast

import jsonschema_rs
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from rfnry_engine.errors import InputError, ModelError
from rfnry_engine.foreground import current_call
from rfnry_engine.models import Part
from rfnry_engine.tools.folder import ToolDef, reason
from rfnry_engine.tools.tools import Result, Tool
from rfnry_engine.tree import AgentSpec
from rfnry_engine.workers.pool import WorkerPool

PARTS = TypeAdapter(list[Part])
PART_TYPES = ("text", "image", "audio", "document")


@cache
def described(args: type[BaseModel]) -> str:
    return json.dumps(args.model_json_schema())


@cache
def _validator(schema: str) -> jsonschema_rs.Validator:
    return jsonschema_rs.validator_for(schema, offline=True)


def _compiled(schema: dict[str, JsonValue]) -> jsonschema_rs.Validator:
    return _validator(json.dumps(schema, sort_keys=True))


def picked(spec: AgentSpec, name: str) -> bool:
    return spec.keep is None or name in spec.keep


def chosen(spec: AgentSpec) -> dict[str, ToolDef]:
    return {name: tool for name, tool in spec.folder_tools.items() if picked(spec, name)}


def _call(scope: str) -> dict[str, str]:
    held = current_call()
    return held.model_dump() if held is not None and held.scope == scope else {"scope": scope}


def _parted(name: str, result: JsonValue) -> Result:
    if not (
        isinstance(result, list)
        and result
        and all(isinstance(item, dict) and item.get("type") in PART_TYPES for item in result)
    ):
        return cast(Result, result)
    try:
        return PARTS.validate_json(json.dumps(result))
    except ValidationError as exc:
        raise ModelError(
            f"declared: {name} returned parts that do not fit the contract: {exc.errors(include_url=False, include_input=False)}"
        ) from exc


def folder_tool(definition: ToolDef, pool: WorkerPool, timeout: float | None, scope: str) -> Tool:
    name = definition.name
    arguments = _compiled(definition.input_schema)
    returns = None if definition.output_schema is None else _compiled(definition.output_schema)

    async def body(args: dict[str, Any]) -> Result:
        try:
            arguments.validate(args)
        except jsonschema_rs.ValidationError as exc:
            raise InputError(f"declared: invalid arguments for {name}: {reason(exc)}") from exc
        result = await pool.call(definition.key, args, _call(scope), timeout)
        if returns is not None:
            try:
                returns.validate(result)
            except jsonschema_rs.ValidationError as exc:
                raise ModelError(
                    f"declared: {name} returned a result that does not match its output_schema: {reason(exc)}"
                ) from exc
        return _parted(name, result)

    return Tool(
        name, definition.description, body=body, input_schema=deepcopy(definition.input_schema)
    )
