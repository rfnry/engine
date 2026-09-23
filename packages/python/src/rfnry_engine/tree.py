from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.frontmatter import parse_frontmatter_doc
from rfnry_engine.models import Check, Choice, Question, Score

Kind = Literal["generation", "decision"]


class RouteSpec(BaseModel):
    by: str
    to: dict[str, str]
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentFront(BaseModel):
    name: str
    kind: Kind
    description: str = ""
    agents: list[str] = Field(default_factory=list)
    route: RouteSpec | None = None
    workspace: bool = False


class QuestionFront(BaseModel):
    type: Literal["choice", "score", "check"]
    options: dict[str, str] | None = None
    levels: list[str] | None = None
    yes: str | None = None
    no: str | None = None


class ToolField(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "object", "array"]
    description: str = ""
    required: bool = False


class ToolFront(BaseModel):
    name: str
    description: str | None = None
    input: dict[str, ToolField] | None = None


class ToolDecl(BaseModel):
    name: str
    description: str | None
    input: dict[str, ToolField] | None


class AgentSpec(BaseModel):
    root: Path
    name: str
    kind: Kind
    description: str
    body: str
    agents: list[str]
    route: RouteSpec | None
    questions: dict[str, Question]
    tools: dict[str, ToolDecl]
    tasks: list[str]
    workspace: bool = False


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _question(path: Path) -> Question:
    front, body = parse_frontmatter_doc(
        _read(path), model=QuestionFront, source=path, label="question"
    )
    instructions = body.strip()
    try:
        if front.type == "choice":
            return Choice(instructions=instructions, options=front.options or {})
        if front.type == "score":
            return Score(instructions=instructions, levels=front.levels or [])
        return Check(instructions=instructions, yes=front.yes, no=front.no)
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}: {exc}") from exc


def _tool_decl(path: Path) -> ToolDecl:
    front, body = parse_frontmatter_doc(
        _read(path), model=ToolFront, source=path, label="tool"
    )
    if front.name != path.stem:
        raise ConfigurationError(f"tree: tool {front.name!r} must live in {front.name}.md: {path}")
    if front.description is None:
        if front.input is not None or body.strip():
            raise ConfigurationError(f"tree: tool file {path} adopts a code tool by name, so it takes no input or description; give it a description to make it a contract")
        return ToolDecl(name=front.name, description=None, input=None)
    description = front.description if not body.strip() else f"{front.description}\n\n{body.strip()}"
    return ToolDecl(name=front.name, description=description, input=dict(front.input or {}))


def _markdown_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".md" and not p.name.startswith(("_", ".")))


def load_agent(directory: Path) -> AgentSpec:
    front, body = parse_frontmatter_doc(
        _read(directory / "AGENT.md"),
        model=AgentFront,
        source=directory / "AGENT.md",
        label="agent",
    )
    if front.name != directory.name:
        raise ConfigurationError(f"tree: agent {front.name!r} must live in a directory named {front.name!r}: {directory}")
    questions = {p.stem: _question(p) for p in _markdown_files(directory / "questions")}
    tools = {p.stem: _tool_decl(p) for p in _markdown_files(directory / "tools")}
    tasks = [p.stem for p in _markdown_files(directory / "tasks")]
    if front.kind == "decision" and not questions:
        raise ConfigurationError(f"tree: decision agent {front.name!r} has no questions/")
    if front.kind == "decision" and tools:
        raise ConfigurationError(f"tree: decision agent {front.name!r} cannot declare tools/")
    if front.kind == "decision" and front.workspace:
        raise ConfigurationError(f"tree: decision agent {front.name!r} cannot declare workspace: true")
    if front.kind == "generation" and questions:
        raise ConfigurationError(f"tree: generation agent {front.name!r} cannot declare questions/")
    if front.kind == "generation" and front.route is not None:
        raise ConfigurationError(f"tree: generation agent {front.name!r} cannot declare route:")
    return AgentSpec(
        root=directory.resolve(),
        name=front.name,
        kind=front.kind,
        description=front.description,
        body=body.strip(),
        agents=front.agents,
        route=front.route,
        questions=questions,
        tools=tools,
        tasks=tasks,
        workspace=front.workspace,
    )


def load_tree(root: Path) -> dict[str, AgentSpec]:
    root = Path(root)
    if (root / "AGENT.md").is_file():
        spec = load_agent(root)
        return {spec.name: spec}
    agents = {
        child.name: load_agent(child)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "AGENT.md").is_file()
    }
    if not agents:
        raise ConfigurationError(f"tree: no agents under {root}")
    return agents


def validate_tree(agents: dict[str, AgentSpec]) -> None:
    for spec in agents.values():
        for target in spec.agents:
            if target not in agents:
                raise ConfigurationError(f"tree: agent {spec.name!r} names unknown agent {target!r}")
        if spec.route is None:
            continue
        question = spec.questions.get(spec.route.by)
        if not isinstance(question, Choice):
            raise ConfigurationError(f"tree: route of {spec.name!r} must name a choice question")
        for option, target in spec.route.to.items():
            if option not in question.options:
                raise ConfigurationError(f"tree: route of {spec.name!r} maps unknown option {option!r}")
            if target not in agents:
                raise ConfigurationError(f"tree: route of {spec.name!r} points at unknown agent {target!r}")
