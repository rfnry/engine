from rfnry_engine.models import ToolSpec
from rfnry_engine.tree import AgentSpec


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
        ("TASK", "" if task is None else spec.task_bodies.get(task, "")),
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
