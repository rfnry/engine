from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import ToolCall
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.declared import declared_tools
from rfnry_engine.tools.delegate import delegate_tool
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
    ) -> None:
        if not scope:
            raise InputError("engine_tools: scope is required")
        spec = engine._spec(agent)
        if spec.workspace and workspace is None:
            raise ConfigurationError(
                f"engine_tools: agent {agent!r} declares workspace: true; pass workspace= to the run"
            )
        room = workspace if spec.workspace else None
        media_types = getattr(engine.generation, "media_types", None)
        items: list[Tool | ToolsLike] = [
            *builtins(spec.root),
            load_tool(spec.root, room.root if room else None, media_types),
            *declared_tools(spec, engine.tools),
        ]
        if room is not None:
            items.extend(room.tools())
        if spec.agents:
            items.append(delegate_tool(engine, spec, scope, tools, workspace, tags, approve))
        if tools is not None:
            items.append(tools)
        super().__init__(items)
