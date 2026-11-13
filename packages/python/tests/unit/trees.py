import shutil
from collections.abc import Iterable, Mapping
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


def solo(
    root: Path,
    *,
    name: str = "a",
    kind: str = "generation",
    model: str = "strong",
    body: str = "Hi.\n",
    tools: Iterable[str] = (),
    written: Mapping[str, str] | None = None,
    declared: Mapping[str, Any] | None = None,
    **entry: Any,
) -> Path:
    agent: dict[str, Any] = {"kind": kind, "model": model, "instructions": f"{name}.md", **entry}
    picked = list(tools)
    if picked:
        agent["tools"] = {**agent.get("tools", {}), "allow": picked}
    return write_folder(
        root, {name: agent}, tools=declared, files={f"{name}.md": body, **(written or {})}
    )


def add_agent(
    root: Path,
    name: str,
    entry: Mapping[str, Any],
    files: Mapping[str, str] | None = None,
    tools: Mapping[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    held: dict[str, Any] = {}
    manifest = root / MANIFEST
    if manifest.is_file():
        held = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    return write_folder(
        root,
        {**held.get("agents", {}), name: entry},
        tools={**held.get("tools", {}), **(tools or {})} or None,
        files=files,
    )


def deciding(
    root: Path,
    name: str,
    questions: Mapping[str, str],
    *,
    route: Mapping[str, Any] | None = None,
    model: str = "judge",
    body: str = "",
    **entry: Any,
) -> Path:
    agent: dict[str, Any] = {
        "kind": "decision",
        "model": model,
        "instructions": f"{name}.md",
        **({"questions": [f"{name}/*.json"]} if questions else {}),
        **entry,
    }
    if route is not None:
        agent["orchestration"] = {"route": dict(route)}
    return add_agent(
        root,
        name,
        agent,
        files={
            f"{name}.md": body,
            **{f"{name}/{qid}.json": text for qid, text in questions.items()},
        },
    )


def amend(root: Path, agent: str, **sections: Any) -> Path:
    document = yaml.safe_load((root / MANIFEST).read_text(encoding="utf-8"))
    entry = document["agents"][agent]
    for key, value in sections.items():
        entry[key] = {**entry[key], **value} if isinstance(entry.get(key), dict) else value
    (root / MANIFEST).write_text(yaml.dump(document, sort_keys=False), encoding="utf-8")
    return root


def tooled(fixtures: Path, tmp_path: Path, agent: str, *tools: str) -> Path:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    document = yaml.safe_load((root / MANIFEST).read_text(encoding="utf-8"))
    held = document["agents"][agent].get("tools", {}).get("allow", [])
    return amend(root, agent, tools={"allow": [*held, *tools]})


def provided(adapters: Mapping[str, Any]) -> dict[str, Provider]:
    def giving(adapter: Any) -> Provider:
        def build(model: str | None, effort: str | None, /) -> Any:
            return adapter

        return build

    return {name: giving(adapter) for name, adapter in adapters.items()}
