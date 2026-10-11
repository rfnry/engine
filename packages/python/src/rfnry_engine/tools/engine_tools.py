from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import HistoryLike, ToolCall
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.declared import declared_tools
from rfnry_engine.tools.delegate import delegate_tool
from rfnry_engine.tools.history import HISTORY_TOOL, history_tool
from rfnry_engine.tools.tools import Tool, Tools, ToolsLike
from rfnry_engine.tools.workspace import Workspace

if TYPE_CHECKING:
    from rfnry_engine.engine import Engine


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
    ) -> None:
        if not scope:
            raise InputError("engine_tools: scope is required")
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
        declared = declared_tools(spec, engine.tools)
        given = [] if tools is None else [d["name"] for d in tools.definitions()]
        if HISTORY_TOOL in [*(t.name for t in declared), *given]:
            raise ConfigurationError(
                f"engine_tools: a tool named {HISTORY_TOOL!r} collides with the builtin"
            )
        room = workspace if spec.workspace else None
        media_types = getattr(engine.generation, "media_types", None)
        items: list[Tool | ToolsLike] = [
            *builtins(spec.root),
            load_tool(spec.root, room.root if room else None, media_types),
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
            items.append(delegate_tool(engine, spec, scope, tools, workspace, tags, approve))
        if tools is not None:
            items.append(tools)
        super().__init__(items)
