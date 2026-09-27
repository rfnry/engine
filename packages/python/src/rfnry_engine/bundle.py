from __future__ import annotations

from pathlib import Path

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import ToolSpec
from rfnry_engine.tree import AgentSpec


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _task(spec: AgentSpec, task: str | None) -> str:
    if task is None:
        return ""
    path = spec.root / "tasks" / f"{task}.md"
    if task not in spec.tasks or not path.is_file():
        raise ConfigurationError(f"bundle: agent {spec.name!r} has no task {task!r}")
    return _read(path)


def build_bundle(
    spec: AgentSpec,
    *,
    task: str | None,
    tools: list[ToolSpec],
    agents: list[AgentSpec],
    context: str | None,
) -> str:
    sections = [
        ("AGENT", spec.body),
        ("INDEX", _read(spec.root / "INDEX.md")),
        ("TASK", _task(spec, task)),
        ("RULES", _read(spec.root / "rules" / "_index.md")),
        ("SKILLS", _read(spec.root / "skills" / "_index.md")),
        (
            "TOOLS",
            "\n".join(
                f"- {t.name}: {t.description.splitlines()[0] if t.description else ''}"
                for t in tools
            ),
        ),
        ("KNOWLEDGE", _read(spec.root / "knowledge" / "_index.md")),
        ("AGENTS", "\n".join(f"- {a.name}: {a.description}" for a in agents)),
        ("CONTEXT", (context or "").strip()),
    ]
    return "\n\n".join(f"<<<{label}>>>\n{body}" for label, body in sections if body)
