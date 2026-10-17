from collections.abc import Callable, Collection, Mapping
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    ValidationError,
    field_validator,
)

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.fingerprint import folder_fingerprint
from rfnry_engine.folders import DECLARED, content, declares, entries, joined, listing
from rfnry_engine.frontmatter import parse_frontmatter_doc, plainly
from rfnry_engine.jail import Jail
from rfnry_engine.models import Check, Choice, Part, Question, Score
from rfnry_engine.tools.builtins import preload_parts

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

    kind: Kind
    description: str = ""
    agents: list[str] = Field(default_factory=list)
    route: RouteSpec | None = None
    workspace: bool = False
    preload: bool = False
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

    description: str | None = None
    input: dict[str, ToolField] | None = None


class ToolsFront(RootModel[dict[str, str | None]]):
    pass


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
    preload: list[Part] = Field(default_factory=list)
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
            f"tree: question {path} gives instructions both in its frontmatter and in its body; keep one"
        )
    instructions = front.instructions or written
    if not instructions:
        raise ConfigurationError(
            f"tree: question {path} gives no instructions, in its body or its frontmatter; write the question under its frontmatter"
        )
    try:
        if front.type == "choice":
            return Choice(instructions=instructions, options=front.options or {})
        if front.type == "score":
            return Score(instructions=instructions, levels=front.levels or [])
        return Check(instructions=instructions, yes=front.yes, no=front.no)
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}; {plainly(exc)}") from exc


def _listed(path: Path) -> dict[str, ToolDecl]:
    if not path.is_file():
        return {}
    front, body = parse_frontmatter_doc(_read(path), model=ToolsFront, source=path, label="tools")
    if body.strip():
        raise ConfigurationError(
            f"tree: {path} lists its tools in its frontmatter; a tool that needs more has a file in tools/"
        )
    blank = [name for name, said in front.root.items() if said is not None and not said.strip()]
    if blank:
        raise ConfigurationError(
            f"tree: {path} gives {', '.join(map(repr, blank))} a blank description; leave it out to use the code's"
        )
    return {
        name: ToolDecl(name=name, description=said, input=None) for name, said in front.root.items()
    }


def _tool_file(path: Path) -> ToolDecl:
    front, body = parse_frontmatter_doc(_read(path), model=ToolFront, source=path, label="tool")
    if front.description is None or not front.description.strip():
        raise ConfigurationError(
            f"tree: tool file {path} gives no description; a tool that only needs its name is listed in TOOLS.md"
        )
    written = body.strip()
    description = f"{front.description}\n\n{written}" if written else front.description
    return ToolDecl(name=path.stem, description=description, input=front.input)


def _tools(directory: Path, agent: str) -> dict[str, ToolDecl]:
    listed = _listed(directory / "TOOLS.md")
    filed = {p.stem: _tool_file(p) for p in _markdown_files(directory / "tools")}
    twice = sorted(name for name in filed if name in listed)
    if twice:
        raise ConfigurationError(
            f"tree: agent {agent!r} declares {', '.join(map(repr, twice))} in both TOOLS.md and tools/; keep one"
        )
    return dict(sorted({**listed, **filed}.items()))


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


class _Folder:
    def __init__(self, place: Path) -> None:
        self.place = place

    @cached_property
    def files(self) -> list[tuple[str, Path]]:
        return content(self.place, False)

    @cached_property
    def preloaded(self) -> list[tuple[str, Path]]:
        return content(self.place, declares(self.place))

    @cached_property
    def mapped(self) -> list[str]:
        return entries(self.place, declares(self.place))


def _reads(
    agent: str, front: AgentFront, directory: Path, folders: Mapping[str, _Folder], whole: bool
) -> dict[str, _Folder]:
    if whole and front.reads:
        raise ConfigurationError(
            f"tree: root agent {agent!r} cannot declare reads:; it holds its whole folder already"
        )
    if front.kind == "decision" and front.reads:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare reads:; a decision agent reads no files, so give it what it needs in its input or its questions"
        )
    for name in front.reads:
        if name.startswith(".") or name == directory.name or name not in folders:
            choices = ", ".join(sorted(n for n in folders if n != directory.name)) or "none"
            raise ConfigurationError(
                f"tree: agent {agent!r} reads unknown folder {name!r}; reads: names a folder beside it in the root, one of: {choices}"
            )
    if front.reads and directory.resolve().parent != directory.parent.resolve():
        raise ConfigurationError(
            f"tree: agent {agent!r} declares reads: but its folder is a link to {directory.resolve()}; reads: needs the agent's folder inside the root, so link the whole root instead"
        )
    return {name: folders[name] for name in front.reads}


def _under(
    reads: Mapping[str, _Folder], files: Callable[[_Folder], list[tuple[str, Path]]]
) -> list[tuple[str, Path]]:
    return [
        (f"../{name}/{rel}", path) for name, folder in reads.items() for rel, path in files(folder)
    ]


def _readable(root: Path, whole: bool = False) -> dict[str, Path]:
    jail = Jail(root)
    found: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.name.startswith(".") or not child.is_dir() or (whole and child.name in DECLARED):
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
    folders: Mapping[str, _Folder] | None = None,
    name: str | None = None,
    sub: Collection[str] = (),
    whole: bool = False,
) -> AgentSpec:
    front, body = parse_frontmatter_doc(
        _read(directory / "AGENT.md"),
        model=AgentFront,
        source=directory / "AGENT.md",
        label="agent",
    )
    agent = name or directory.resolve().name
    questions = _questions(directory / "questions")
    tools = _tools(directory, agent)
    tasks = [p.stem for p in _markdown_files(directory / "tasks")]
    task_questions: dict[str, dict[str, Question]] = {}
    if front.kind == "decision":
        if tasks:
            raise ConfigurationError(
                f"tree: decision agent {agent!r} has tasks/{tasks[0]}.md, but keeps each task's questions in a folder; move it to tasks/{tasks[0]}/"
            )
        task_questions = {p.name: _questions(p) for p in _folders(directory / "tasks")}
        tasks = list(task_questions)
    for task, asked in task_questions.items():
        clash = [qid for qid in asked if qid in questions]
        if clash:
            raise ConfigurationError(
                f"tree: decision agent {agent!r} asks {', '.join(map(repr, clash))} in both questions/ and tasks/{task}/; rename one"
            )
    if front.kind == "decision" and tools:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare tools; tools are for generation agents, so give them to one it routes to"
        )
    if front.kind == "decision" and front.workspace:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare workspace: true; a workspace is for generation agents"
        )
    if front.kind == "decision" and front.preload:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare preload:; a decision agent reads no files, so give it what it needs in its input or its questions"
        )
    if front.kind == "generation" and questions:
        raise ConfigurationError(
            f"tree: generation agent {agent!r} cannot declare questions/; questions are for decision agents, so make it kind: decision or move them to one"
        )
    if front.kind == "generation" and front.route is not None:
        raise ConfigurationError(
            f"tree: generation agent {agent!r} cannot declare route:; routes are for decision agents"
        )
    reads = _reads(agent, front, directory, folders or {}, whole)
    root = directory.resolve()
    files = sorted([*content(root, False), *_under(reads, lambda folder: folder.files)])
    try:
        preload, read = (
            preload_parts(
                sorted(
                    rel
                    for rel, _ in [
                        *content(root, True, sub),
                        *_under(reads, lambda folder: folder.preloaded),
                    ]
                ),
                media_types,
                Jail(root, also=[folder.place for folder in reads.values()]),
            )
            if front.preload
            else ([], {})
        )
    except InputError as exc:
        raise ConfigurationError(
            f"tree: agent {agent!r} cannot preload its folder: {exc}; move that file out of its folders, or drop preload: true"
        ) from exc
    return AgentSpec(
        root=root,
        name=agent,
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
        preload=preload,
        listing=joined(
            entries(root, True, sub), {name: folder.mapped for name, folder in reads.items()}
        ),
        reads={name: folder.place for name, folder in reads.items()},
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
        fingerprint=folder_fingerprint(content(root, False), {}),
        listing=listing(root, False, {}),
        plain=True,
    )


def load_tree(
    root: Path, media_types: Collection[str] | None = None, name: str | None = None
) -> dict[str, AgentSpec]:
    root = Path(root)
    whole = (root / "AGENT.md").is_file()
    folders = {child: _Folder(place) for child, place in _readable(root, whole).items()}
    agents = {
        child.name: load_agent(child, media_types, folders, child.name)
        for child in sorted(root.iterdir())
        if child.is_dir()
        and not child.name.startswith(".")
        and not (whole and child.name in DECLARED)
        and (child / "AGENT.md").is_file()
    }
    if whole:
        spec = load_agent(root, media_types, {}, name, sub=set(agents), whole=True)
        if spec.name in agents:
            fix = (
                "rename that folder, or pass Engine(name=) another name"
                if name is not None
                else "rename the folder, or pass Engine(name=) for the root"
            )
            raise ConfigurationError(
                f"tree: the root agent and the agent in {root / spec.name} are both named {spec.name!r}; {fix}"
            )
        return {spec.name: spec, **agents}
    if agents:
        if name is not None:
            raise ConfigurationError(
                f"tree: name= names the root's own agent, and {root} has no AGENT.md, only agents in its folders; drop name=, or give the root an AGENT.md"
            )
        return agents
    spec = plain_agent(root, name or root.resolve().name)
    return {spec.name: spec}


def validate_tree(agents: dict[str, AgentSpec]) -> None:
    for spec in agents.values():
        for name in spec.agents:
            if name not in agents:
                raise ConfigurationError(
                    f"tree: agent {spec.name!r} names unknown agent {name!r}; agents: names agents of this tree: {', '.join(sorted(agents))}"
                )
        if spec.route is None:
            continue
        question = spec.questions.get(spec.route.by)
        if not isinstance(question, Choice):
            raise ConfigurationError(
                f"tree: route of {spec.name!r} must name a choice question; by: names a choice question in its questions/"
            )
        for option, target in spec.route.to.items():
            if option not in question.options:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} maps unknown option {option!r}; the options of {spec.route.by!r} are {', '.join(question.options)}"
                )
            routed = agents.get(target.agent)
            if routed is None:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} points at unknown agent {target.agent!r}; the tree has {', '.join(sorted(agents))}"
                )
            if target.task is not None and target.task not in routed.tasks:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} points at agent {target.agent!r}, which has no task {target.task!r}; its tasks are {', '.join(routed.tasks) or 'none'}"
                )
            tasked = None if target.task is None else routed.task_questions.get(target.task)
            if routed.kind == "decision" and not routed.questions and not tasked:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} gives decision agent {target.agent!r} nothing to ask: it has no questions/ and the route names no task with questions; give it questions/, or name one of its tasks in the route"
                )
