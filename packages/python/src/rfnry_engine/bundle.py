from rfnry_engine.models import ToolSpec
from rfnry_engine.tree import AgentSpec


def build_bundle(
    spec: AgentSpec,
    *,
    tools: list[ToolSpec],
    agents: list[AgentSpec],
    context: str | None,
) -> str:
    opens = any(t.name in ("Read", "Load") for t in tools)
    sections = [
        ("AGENT", spec.body),
        ("FILES", spec.listing if opens else ""),
        (
            "SKILLS",
            "\n".join(
                f"{s.path} — {s.description}" if s.description else s.path
                for name, s in spec.skills.items()
                if name in spec.offered
            )
            if opens
            else "",
        ),
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
