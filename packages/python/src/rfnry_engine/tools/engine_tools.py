import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import HistoryLike, ToolCall
from rfnry_engine.records import EventHandler
from rfnry_engine.stop import Stop
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.declared import declared_tools
from rfnry_engine.tools.delegate import delegate_tool
from rfnry_engine.tools.history import HISTORY_TOOL, history_tool
from rfnry_engine.tools.tools import Tool, Tools, ToolsLike
from rfnry_engine.tools.workspace import Workspace
from rfnry_engine.tree import from_agent

if TYPE_CHECKING:
    from rfnry_engine.engine import Engine


@cache
def _described(args: type[BaseModel]) -> str:
    return json.dumps(args.model_json_schema())


def _own(item: Tool | ToolsLike) -> Tool | ToolsLike:
    if isinstance(item, Tool) and item.args is not None and item.input_schema is None:
        return replace(item, input_schema=json.loads(_described(item.args)))
    return item


class EngineTools(Tools):
    def __init__(
        self,
        engine: Engine,
        *,
        agent: str,
        scope: str,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
        history: Sequence[object] | HistoryLike | None = None,
        start: int = 0,
        on_event: EventHandler | None = None,
        stop: Callable[[], Stop | None] | None = None,
    ) -> None:
        if not scope:
            raise InputError("engine_tools: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"engine_tools: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        if start < 0:
            raise InputError(f"engine_tools: start must not be negative; got {start}")
        if start and history is None:
            raise InputError("engine_tools: start needs a history")
        if history is not None and not isinstance(history, HistoryLike):
            history = list(history)
            if start > len(history):
                raise InputError(
                    f"engine_tools: start must be between 0 and {len(history)}; got {start}"
                )
        spec = engine._spec(agent)
        if spec.workspace and workspace is None:
            raise ConfigurationError(
                f"engine_tools: agent {agent!r} declares workspace: true; pass workspace= to the run"
            )
        declared = [
            replace(tool, input_schema=deepcopy(tool.input_schema))
            for tool in engine._declared.get(agent) or declared_tools(spec, engine.tools)
        ]
        given = [] if tools is None else [d["name"] for d in tools.definitions()]
        if HISTORY_TOOL in [*(t.name for t in declared), *given]:
            raise ConfigurationError(
                f"engine_tools: a tool named {HISTORY_TOOL!r} collides with the builtin"
            )
        room = workspace if spec.workspace else None
        media_types = getattr(engine.generation, "media_types", None)
        backend = room.backend if room else None
        items: list[Tool | ToolsLike] = [
            *builtins(spec.root, from_agent(spec.path, spec.reads), backend),
            load_tool(spec.root, backend, media_types, spec.reads.values()),
        ]
        if history is not None and start > 0:
            items.append(
                history_tool(
                    history,
                    start,
                    walk=engine.history_walk,
                    chars=engine.history_chars,
                    timeout=engine.tool_timeout,
                )
            )
        items.extend(declared)
        if room is not None:
            items.extend(room.tools())
        if spec.agents:
            items.append(
                delegate_tool(engine, spec, scope, tools, workspace, tags, approve, on_event, stop)
            )
        if tools is not None:
            items.append(tools)
        super().__init__(map(_own, items))
