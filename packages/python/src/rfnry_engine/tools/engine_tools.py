import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import HistoryLike, ToolCall
from rfnry_engine.records import EventHandler
from rfnry_engine.stop import Stop
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.declared import chosen, described, folder_tool, picked
from rfnry_engine.tools.delegate import delegate_tool
from rfnry_engine.tools.history import HISTORY_TOOL, history_tool
from rfnry_engine.tools.tools import Tool, Tools, ToolsLike
from rfnry_engine.tools.workspace import Workspace
from rfnry_engine.tree import from_agent

if TYPE_CHECKING:
    from rfnry_engine.engine import Engine


def _own(item: Tool | ToolsLike) -> Tool | ToolsLike:
    if isinstance(item, Tool) and item.args is not None and item.input_schema is None:
        return replace(item, input_schema=json.loads(described(item.args)))
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
        given = Tools() if tools is None else Tools([tools])
        if HISTORY_TOOL in given:
            raise ConfigurationError(
                f"engine_tools: a tool named {HISTORY_TOOL!r} collides with the builtin"
            )
        room = workspace if spec.workspace else None
        media_types = getattr(engine.models.get(spec.model), "media_types", None)
        backend = room.backend if room else None
        files = [
            *builtins(spec.root, from_agent(spec.path, spec.reads), backend),
            load_tool(spec.root, backend, media_types, spec.reads.values()),
        ]
        items = [tool for tool in files if picked(spec, tool.name)]
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
        pool = engine._workers.pool(agent)
        folder = [
            folder_tool(definition, pool, engine.tool_timeout, scope)
            for name, definition in chosen(spec).items()
            if name not in given
        ]
        items.extend(folder)
        if room is not None:
            items.extend(room.tools())
        if spec.agents:
            items.append(
                delegate_tool(engine, spec, scope, tools, workspace, tags, approve, on_event, stop)
            )
        kept: list[Tool | ToolsLike] = [tool for tool in items if tool.name not in spec.drop]
        super().__init__(map(_own, [*kept, given.without(*spec.drop)]))
        self._in_workers = frozenset(tool.name for tool in folder)
