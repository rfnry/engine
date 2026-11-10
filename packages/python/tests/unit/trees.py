from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from rfnry_engine.manifest import MANIFEST
from rfnry_engine.models import Provider


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _picked_models(agents: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    names: list[Any] = []
    for agent in agents.values():
        if not isinstance(agent, Mapping):
            continue
        modes = agent.get("modes")
        layers = [agent, *(modes.values() if isinstance(modes, Mapping) else [])]
        names.extend(layer.get("model") for layer in layers if isinstance(layer, Mapping))
    return {name: {"provider": name} for name in names if isinstance(name, str) and name.strip()}


def write_folder(
    root: Path,
    agents: Mapping[str, Mapping[str, Any]],
    *,
    workers: Mapping[str, Any] | None = None,
    models: Mapping[str, Any] | None = None,
    tools: Mapping[str, Any] | None = None,
    files: Mapping[str, str] | None = None,
) -> Path:
    document: dict[str, Any] = {"version": 1}
    if workers is not None:
        document["workers"] = workers
    declared = _picked_models(agents) if models is None else models
    if declared or models is not None:
        document["models"] = declared
    if tools is not None:
        document["tools"] = tools
    document["agents"] = agents
    (root / MANIFEST).write_text(
        yaml.dump(document, Dumper=_NoAliases, sort_keys=False), encoding="utf-8"
    )
    for name, text in (files or {}).items():
        place = root / name
        place.parent.mkdir(parents=True, exist_ok=True)
        place.write_text(text, encoding="utf-8")
    return root


def provided(adapters: Mapping[str, Any]) -> dict[str, Provider]:
    def giving(adapter: Any) -> Provider:
        def build(model: str | None, effort: str | None, /) -> Any:
            return adapter

        return build

    return {name: giving(adapter) for name, adapter in adapters.items()}
