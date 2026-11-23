import json
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Required, TypedDict, Unpack

from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import HistoryLike, Refused, RunModel, ToolCall
from rfnry_engine.records import EventHandler
from rfnry_engine.stop import Stop
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.declared import chosen, described, folder_tool, smallest
from rfnry_engine.tools.delegate import delegate_tool
from rfnry_engine.tools.helpers import HELPER_TOOLS
from rfnry_engine.tools.history import HISTORY_TOOL, history_tool
from rfnry_engine.tools.tools import Tool, Tools, ToolsLike
from rfnry_engine.tools.workspace import Workspace
from rfnry_engine.tree import AgentSpec, no_such_skill

if TYPE_CHECKING:
    from rfnry_engine.engine import Engine

HistorySource = tuple[Sequence[object] | HistoryLike, Callable[[], int]]


def _own(item: Tool | ToolsLike) -> Tool | ToolsLike:
    if isinstance(item, Tool) and item.args is not None and item.input_schema is None:
        return replace(item, input_schema=json.loads(described(item.args)))
    return item


def refuse_builtin(tools: ToolsLike | None, name: str) -> None:
    if tools is not None and name in Tools([tools]):
        raise ConfigurationError(
            f"tools/engine_tools: a tool named {name!r} collides with the builtin"
        )


def _withheld_on(spec: AgentSpec, invoked: str | None) -> Callable[[Path], bool]:
    if invoked is None:
        return spec.withheld
    return spec.withheld_when_invoked.get(invoked, spec.withheld)


class EngineTools(Tools):
    def __init__(
        self,
        engine: Engine,
        *,
        agent: str,
        scope: str,
        mode: str | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Callable[[str, ToolCall], Awaitable[bool | Refused]] | None = None,
        history: Sequence[object] | HistoryLike | None = None,
        start: int = 0,
        on_event: EventHandler | None = None,
        stop: Callable[[], Stop | None] | None = None,
        invoked: str | None = None,
        models: Mapping[str, RunModel] | None = None,
    ) -> None:
        engine._open()
        if not scope:
            raise InputError("tools/engine_tools: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"tools/engine_tools: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        if start < 0:
            raise InputError(f"tools/engine_tools: start must not be negative; got {start}")
        if start and history is None:
            raise InputError("tools/engine_tools: start needs a history")
        if history is not None and not isinstance(history, HistoryLike):
            history = list(history)
            if start > len(history):
                raise InputError(
                    f"tools/engine_tools: start must be between 0 and {len(history)}; got {start}"
                )
        spec = engine._spec(agent, mode)
        if invoked is not None and invoked not in spec.skills:
            raise NotFoundError(f"tools/engine_tools: {no_such_skill(spec, invoked)}")
        if spec.workspace and workspace is None:
            raise ConfigurationError(
                f"tools/engine_tools: agent {agent!r} declares workspace: true; pass workspace= to the run"
            )
        refuse_builtin(tools, HISTORY_TOOL)
        given = Tools() if tools is None else Tools([tools])
        offered = sorted({*spec.granted, *spec.folder_tools, *given.names})
        for name in spec.keep:
            if name not in offered:
                raise ConfigurationError(
                    f"tools/engine_tools: agent {agent!r} lists {name!r} in tools.allow, which this run does not offer; its tools are {', '.join(offered) or 'none'}"
                )
        given = given.without(*(name for name in given.names if name not in spec.keep))
        clash = [name for name in sorted(HELPER_TOOLS) if name in given]
        if spec.background and clash:
            raise ConfigurationError(f"tools/engine_tools: duplicate tool name {clash[0]!r}")
        room = workspace if spec.workspace else None
        merged = engine._run_models(models)

        def media_types() -> Collection[str] | None:
            return getattr(engine._built(merged.get(spec.model)), "media_types", None)

        backend = room.backend if room else None
        withheld = _withheld_on(spec, invoked)
        files = [
            *builtins(spec.root, backend, withheld, spec.named),
            load_tool(spec.root, backend, media_types, withheld, spec.named),
        ]
        items: list[Tool | ToolsLike] = [tool for tool in files if tool.name in spec.keep]
        deadline = smallest(spec.tool_timeout, engine.tool_timeout)
        source = self._history_source(history, start)
        if source is not None:
            items.append(
                history_tool(
                    *source,
                    walk=engine.history_walk,
                    chars=engine.history_chars,
                    timeout=deadline,
                )
            )
        pool = engine._workers.pool(spec)
        folder = [
            folder_tool(definition, pool, deadline, scope, dict(tags or {}))
            for name, definition in chosen(spec).items()
            if name not in given
        ]
        items.extend(folder)
        if room is not None:
            items.extend(tool for tool in room.tools() if tool.name in spec.granted)
        if spec.agents:
            items.append(
                delegate_tool(
                    engine, spec, scope, tools, workspace, tags, approve, on_event, stop, merged
                )
            )
        super().__init__(map(_own, [*items, given]))
        self._in_workers = frozenset(tool.name for tool in folder)

    def _history_source(
        self, history: Sequence[object] | HistoryLike | None, start: int
    ) -> HistorySource | None:
        if history is None or start <= 0:
            return None
        return history, lambda: start


class Offered(TypedDict, total=False):
    agent: Required[str]
    scope: Required[str]
    mode: str | None
    tools: ToolsLike | None
    workspace: Workspace | None
    tags: Mapping[str, str] | None
    approve: Callable[[str, ToolCall], Awaitable[bool | Refused]] | None
    history: Sequence[object] | HistoryLike | None
    start: int
    on_event: EventHandler | None
    stop: Callable[[], Stop | None] | None
    invoked: str | None
    models: Mapping[str, RunModel] | None


class RunTools(EngineTools):
    def __init__(
        self, engine: Engine, *, source: HistorySource | None, **offered: Unpack[Offered]
    ) -> None:
        self._source = source
        super().__init__(engine, **offered)

    def _history_source(
        self, history: Sequence[object] | HistoryLike | None, start: int
    ) -> HistorySource | None:
        if self._source is None:
            return super()._history_source(history, start)
        return self._source
