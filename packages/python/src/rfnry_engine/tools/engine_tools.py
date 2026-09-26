from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import ToolCall
from rfnry_engine.tools.builtins import builtins
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
        spec = engine._spec(agent)
        items: list[Tool | ToolsLike] = [*builtins(spec.root), *declared_tools(spec, engine.tools)]
        if spec.workspace:
            if workspace is None:
                raise ConfigurationError(f"engine_tools: agent {agent!r} declares workspace: true; pass workspace= to the run")
            items.extend(workspace.tools())
        if spec.agents:
            items.append(delegate_tool(engine, spec, scope, tools, workspace, tags, approve))
        if tools is not None:
            items.append(tools)
        super().__init__(items)
