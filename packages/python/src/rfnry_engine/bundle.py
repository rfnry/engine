from pathlib import Path

from rfnry_engine.frontmatter import split_frontmatter
from rfnry_engine.models import ToolSpec
from rfnry_engine.tree import AgentSpec


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip() if path.is_file() else ""


def read_body(path: Path) -> str:
    text = _read(path)
    split = split_frontmatter(text)
    return text if split is None else split[1].strip()


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
        ("TASK", "" if task is None else read_body(spec.root / "tasks" / f"{task}.md")),
        ("FILES", spec.listing),
        (
            "TOOLS",
            "\n".join(
                f"- {t.name}: {t.description.splitlines()[0] if t.description else ''}"
                for t in tools
            ),
        ),
        ("AGENTS", "\n".join(f"- {a.name}: {a.description}" for a in agents)),
        ("CONTEXT", (context or "").strip()),
    ]
    return "\n\n".join(f"<<<{label}>>>\n{body}" for label, body in sections if body)
