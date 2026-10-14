from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.fingerprint import folder_files, folder_fingerprint
from rfnry_engine.folders import content, declares, listing
from rfnry_engine.frontmatter import parse_frontmatter_doc
from rfnry_engine.jail import Jail
from rfnry_engine.models import Check, Choice, Part, Question, Score
from rfnry_engine.tools.builtins import boot_parts

Kind = Literal["generation", "decision"]


class RouteTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    task: str | None = None


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
    to: dict[str, RouteTarget]
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("to", mode="before")
    @classmethod
    def _a_name_is_an_agent(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return {
            option: {"agent": target} if isinstance(target, str) else target
            for option, target in value.items()
        }


class AgentFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Kind
    description: str = ""
    agents: list[str] = Field(default_factory=list)
    route: RouteSpec | None = None
    workspace: bool = False
    boot: Literal["all"] | None = None
    reads: list[str] = Field(default_factory=list)


class QuestionFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "score", "check"]
    instructions: dict[str, JsonValue] | list[JsonValue] | None = None
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None] | None = None
    levels: list[str | dict[str, JsonValue] | list[JsonValue]] | None = None
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None


class ToolField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean", "object", "array"]
    description: str = ""
    required: bool = False


class ToolFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    fingerprint: str
    workspace: bool = False
    task_questions: dict[str, dict[str, Question]] = Field(default_factory=dict)
    boot: list[Part] = Field(default_factory=list)
    listing: str = ""
    reads: dict[str, Path] = Field(default_factory=dict)
    plain: bool = False


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _question(path: Path) -> Question:
    front, body = parse_frontmatter_doc(
        _read(path), model=QuestionFront, source=path, label="question"
    )
    written = body.strip()
    if front.instructions is not None and written:
        raise ConfigurationError(
            f"tree: question {path} gives instructions both in its frontmatter and in its body"
        )
    instructions = front.instructions or written
    if not instructions:
        raise ConfigurationError(
            f"tree: question {path} gives no instructions, in its body or its frontmatter"
        )
    try:
        if front.type == "choice":
            return Choice(instructions=instructions, options=front.options or {})
        if front.type == "score":
            return Score(instructions=instructions, levels=front.levels or [])
        return Check(instructions=instructions, yes=front.yes, no=front.no)
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}: {exc}") from exc


def _tool_decl(path: Path) -> ToolDecl:
    front, body = parse_frontmatter_doc(_read(path), model=ToolFront, source=path, label="tool")
    if front.name != path.stem:
        raise ConfigurationError(f"tree: tool {front.name!r} must live in {front.name}.md: {path}")
    if front.description is None:
        if front.input is not None or body.strip():
            raise ConfigurationError(
                f"tree: tool file {path} adopts a code tool by name, so it takes no input or description; give it a description to make it a contract"
            )
        return ToolDecl(name=front.name, description=None, input=None)
    description = (
        front.description if not body.strip() else f"{front.description}\n\n{body.strip()}"
    )
    return ToolDecl(name=front.name, description=description, input=dict(front.input or {}))


def _markdown_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.suffix == ".md" and not p.name.startswith(("_", "."))
    )


def _folders(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    )


def _questions(directory: Path) -> dict[str, Question]:
    return {p.stem: _question(p) for p in _markdown_files(directory)}


def _reads(front: AgentFront, directory: Path, folders: Mapping[str, Path]) -> dict[str, Path]:
    if front.kind == "decision" and front.reads:
        raise ConfigurationError(f"tree: decision agent {front.name!r} cannot declare reads:")
    for name in front.reads:
        if name.startswith(".") or name == directory.name or name not in folders:
            raise ConfigurationError(f"tree: agent {front.name!r} reads unknown folder {name!r}")
    return {name: folders[name] for name in front.reads}


def _under(
    reads: Mapping[str, Path], files: Callable[[Path], list[tuple[str, Path]]]
) -> list[tuple[str, Path]]:
    return [
        (f"../{name}/{rel}", path) for name, folder in reads.items() for rel, path in files(folder)
    ]


def _readable(root: Path) -> dict[str, Path]:
    jail = Jail(root)
    found: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        try:
            place = jail.resolve(child)
        except InputError:
            continue
        if place != jail.root:
            found[child.name] = place
    return found


def load_agent(
    directory: Path,
    media_types: Collection[str] | None = None,
    folders: Mapping[str, Path] | None = None,
) -> AgentSpec:
    front, body = parse_frontmatter_doc(
        _read(directory / "AGENT.md"),
        model=AgentFront,
        source=directory / "AGENT.md",
        label="agent",
    )
    if front.name != directory.name:
        raise ConfigurationError(
            f"tree: agent {front.name!r} must live in a directory named {front.name!r}: {directory}"
        )
    questions = _questions(directory / "questions")
    tools = {p.stem: _tool_decl(p) for p in _markdown_files(directory / "tools")}
    tasks = [p.stem for p in _markdown_files(directory / "tasks")]
    task_questions: dict[str, dict[str, Question]] = {}
    if front.kind == "decision":
        if tasks:
            raise ConfigurationError(
                f"tree: decision agent {front.name!r} keeps each task's questions in tasks/<task>/; got tasks/{tasks[0]}.md"
            )
        task_questions = {p.name: _questions(p) for p in _folders(directory / "tasks")}
        tasks = list(task_questions)
    for task, asked in task_questions.items():
        clash = [qid for qid in asked if qid in questions]
        if clash:
            raise ConfigurationError(
                f"tree: decision agent {front.name!r} asks {', '.join(map(repr, clash))} in both questions/ and tasks/{task}/"
            )
    if front.kind == "decision" and tools:
        raise ConfigurationError(f"tree: decision agent {front.name!r} cannot declare tools/")
    if front.kind == "decision" and front.workspace:
        raise ConfigurationError(
            f"tree: decision agent {front.name!r} cannot declare workspace: true"
        )
    if front.kind == "decision" and front.boot is not None:
        raise ConfigurationError(f"tree: decision agent {front.name!r} cannot declare boot: all")
    if front.kind == "generation" and questions:
        raise ConfigurationError(f"tree: generation agent {front.name!r} cannot declare questions/")
    if front.kind == "generation" and front.route is not None:
        raise ConfigurationError(f"tree: generation agent {front.name!r} cannot declare route:")
    reads = _reads(front, directory, folders or {})
    root = directory.resolve()
    files = sorted([*folder_files(root), *_under(reads, folder_files)])
    shared = _under(reads, lambda folder: content(folder, declares(folder)))
    try:
        boot, read = (
            boot_parts(sorted([*content(root, True), *shared]), media_types)
            if front.boot == "all"
            else ([], {})
        )
    except InputError as exc:
        raise ConfigurationError(
            f"tree: agent {front.name!r} cannot boot its folder: {exc}"
        ) from exc
    return AgentSpec(
        root=root,
        name=front.name,
        kind=front.kind,
        description=front.description,
        body=body.strip(),
        agents=front.agents,
        route=front.route,
        questions=questions,
        tools=tools,
        tasks=tasks,
        fingerprint=folder_fingerprint(files, read),
        workspace=front.workspace,
        task_questions=task_questions,
        boot=boot,
        listing=listing(root, True, reads),
        reads=reads,
    )


def plain_agent(root: Path, name: str) -> AgentSpec:
    root = root.resolve()
    return AgentSpec(
        root=root,
        name=name,
        kind="generation",
        description="",
        body="",
        agents=[],
        route=None,
        questions={},
        tools={},
        tasks=[],
        fingerprint=folder_fingerprint(folder_files(root), {}),
        listing=listing(root, False, {}),
        plain=True,
    )


def _unnamed(name: str | None, root: Path) -> None:
    if name is not None:
        raise ConfigurationError(f"tree: name= names a root with no AGENT.md; {root} has agents")


def load_tree(
    root: Path, media_types: Collection[str] | None = None, name: str | None = None
) -> dict[str, AgentSpec]:
    root = Path(root)
    if (root / "AGENT.md").is_file():
        _unnamed(name, root)
        spec = load_agent(root, media_types)
        return {spec.name: spec}
    folders = _readable(root)
    agents = {
        child.name: load_agent(child, media_types, folders)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "AGENT.md").is_file()
    }
    if agents:
        _unnamed(name, root)
        return agents
    spec = plain_agent(root, name or root.resolve().name)
    return {spec.name: spec}


def validate_tree(agents: dict[str, AgentSpec]) -> None:
    for spec in agents.values():
        for name in spec.agents:
            if name not in agents:
                raise ConfigurationError(f"tree: agent {spec.name!r} names unknown agent {name!r}")
        if spec.route is None:
            continue
        question = spec.questions.get(spec.route.by)
        if not isinstance(question, Choice):
            raise ConfigurationError(f"tree: route of {spec.name!r} must name a choice question")
        for option, target in spec.route.to.items():
            if option not in question.options:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} maps unknown option {option!r}"
                )
            routed = agents.get(target.agent)
            if routed is None:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} points at unknown agent {target.agent!r}"
                )
            if target.task is not None and target.task not in routed.tasks:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} points at agent {target.agent!r}, which has no task {target.task!r}"
                )
            tasked = None if target.task is None else routed.task_questions.get(target.task)
            if routed.kind == "decision" and not routed.questions and not tasked:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} gives decision agent {target.agent!r} nothing to ask: it has no questions/ and the route names no task with questions"
                )
