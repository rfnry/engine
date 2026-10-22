import os
import posixpath
from collections.abc import Callable, Collection, Mapping
from functools import cached_property
from itertools import combinations
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

from rfnry_engine.chain import preload_digest
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.fingerprint import folder_fingerprint
from rfnry_engine.folders import DECLARED, content, declares, entries, joined
from rfnry_engine.frontmatter import (
    body_of,
    description,
    parse_frontmatter_doc,
    plainly,
    split_frontmatter,
)
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

    name: str
    kind: Kind
    model: str
    description: str = ""
    agents: list[str] = Field(default_factory=list)
    route: RouteSpec | None = None
    workspace: bool = False
    preload: bool = False
    reads: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _a_name_is_one_plain_word(cls, value: str) -> str:
        if (
            not value.strip()
            or value != value.strip()
            or value.startswith(".")
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(
                f"tree: name must not be blank, hidden, padded or hold a slash, got {value!r}; write a plain name such as name: support"
            )
        return value

    @field_validator("model")
    @classmethod
    def _a_model_is_plain_text(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(
                f"tree: model must not be blank or padded, got {value!r}; write the name of a model the engine is given, such as model: strong"
            )
        return value


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


class TaskFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None


class ToolDecl(BaseModel):
    name: str
    description: str
    input: dict[str, ToolField] | None


class Skill(BaseModel):
    path: str
    description: str
    body: str


class AgentSpec(BaseModel):
    root: Path
    name: str
    path: str
    kind: Kind
    model: str
    description: str
    body: str
    agents: list[str]
    route: RouteSpec | None
    questions: dict[str, Question]
    tools: dict[str, ToolDecl]
    tasks: dict[str, str]
    task_bodies: dict[str, str] = Field(default_factory=dict)
    fingerprint: str
    workspace: bool = False
    task_questions: dict[str, dict[str, Question]] = Field(default_factory=dict)
    preload: list[Part] = Field(default_factory=list)
    preload_digest: str
    listing: str = ""
    reads: dict[str, Path] = Field(default_factory=dict)
    skills: dict[str, Skill] = Field(default_factory=dict)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"tree: {path} is not UTF-8 text; save it as UTF-8") from exc


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
    bare = [name for name, said in front.root.items() if said is None]
    if bare:
        name = bare[0]
        raise ConfigurationError(
            f"tree: {path} lists {name!r} without a description; write it as {name}: what it does, or give it a file in tools/"
        )
    blank = [name for name, said in front.root.items() if said is not None and not said.strip()]
    if blank:
        raise ConfigurationError(
            f"tree: {path} gives {', '.join(map(repr, blank))} a blank description; write what it does"
        )
    return {
        name: ToolDecl(name=name, description=said, input=None)
        for name, said in front.root.items()
        if said is not None
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


def _task(path: Path) -> tuple[str, str]:
    raw = _read(path)
    if split_frontmatter(raw) is None:
        return "", body_of(raw)
    front, _ = parse_frontmatter_doc(raw, model=TaskFront, source=path, label="task")
    return " ".join((front.description or "").split()), body_of(raw)


def _skills(files: list[tuple[str, Path]], agent: str) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for rel, path in files:
        match rel.split("/"):
            case ["skills", file] if file.endswith(".md"):
                name = file.removesuffix(".md")
            case ["skills", name, "SKILL.md"]:
                pass
            case _:
                continue
        if name.startswith(("_", ".")):
            continue
        if name in found:
            raise ConfigurationError(
                f"tree: agent {agent!r} has the skill {name!r} as both skills/{name}.md and skills/{name}/SKILL.md; keep one"
            )
        raw = _read(path)
        found[name] = Skill(path=rel, description=description(raw, path) or "", body=body_of(raw))
    return dict(sorted(found.items()))


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
        return content(self.place, False, inner=True)

    @cached_property
    def preloaded(self) -> list[tuple[str, Path]]:
        return content(self.place, declares(self.place))

    @cached_property
    def mapped(self) -> list[str]:
        return entries(self.place, declares(self.place))


def from_agent[T](path: str, reads: Mapping[str, T]) -> dict[str, T]:
    return {posixpath.relpath(name, path): place for name, place in reads.items()}


def _linked(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _is_folder(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def _subfolders(folder: Path) -> list[Path]:
    try:
        with os.scandir(folder) as entries:
            return sorted(Path(entry.path) for entry in entries if _is_folder(entry))
    except OSError:
        return []


def _loops(top: Path, found: Mapping[str, Path], key: str, target: Path) -> bool:
    parts = key.split("/")
    above = [top, *(found["/".join(parts[:n])] for n in range(1, len(parts)))]
    return any(place.is_relative_to(target) for place in above)


def _tree_folders(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    top = root.resolve()
    found: dict[str, Path] = {}
    links: dict[str, Path] = {}

    def walk(folder: Path, at: str) -> None:
        holds = declares(folder)
        for child in _subfolders(folder):
            if child.name.startswith(".") or (holds and child.name in DECLARED):
                continue
            place = child.resolve()
            key = f"{at}/{child.name}" if at else child.name
            if _linked(child):
                links[key] = place
                if place.is_relative_to(top) or _loops(top, found, key, place):
                    continue
            found[key] = place
            walk(child, key)

    walk(root, "")
    return found, links


def _holds_agent(folder: Path) -> bool:
    return declares(folder) or any(
        _holds_agent(child)
        for child in _subfolders(folder)
        if not child.name.startswith(".") and not _linked(child)
    )


def _refuse_links(root: Path, found: Mapping[str, Path], links: Mapping[str, Path]) -> None:
    top = root.resolve()
    for key, target in sorted(links.items()):
        inside = target.is_relative_to(top)
        shown = root / target.relative_to(top) if inside else target
        if _loops(top, found, key, target):
            raise ConfigurationError(
                f"tree: {root / key} links to {shown}, a folder that holds the link; remove the link, since the tree would lead into itself"
            )
        if inside and _holds_agent(target):
            raise ConfigurationError(
                f"tree: {root / key} links to {shown}, a folder of the root that holds an AGENT.md; remove the link, since an agent lives in one folder and is called by its name: from anywhere"
            )
        twice = sorted(other for other, place in found.items() if place == target and other != key)
        if not inside and twice and _holds_agent(target):
            raise ConfigurationError(
                f"tree: {root / key} and {root / twice[0]} both reach {shown}, which holds an AGENT.md; remove one of the links, since an agent lives in one folder and is called by its name: from anywhere"
            )


def _written(agent: str, name: str) -> str:
    plain = name.removesuffix("/")
    segments = plain.split("/")
    if (
        not plain
        or plain.startswith("/")
        or "\\" in plain
        or any(not s or s.startswith(".") for s in segments)
    ):
        raise ConfigurationError(
            f"tree: agent {agent!r} reads {name!r}, which is not a path from the root; write it as shared/books, with no leading ./ or /, no .. or hidden folder, no empty segment and no backslash"
        )
    return plain


def _reads(
    agent: str,
    front: AgentFront,
    directory: Path,
    root: Path,
    path: str,
    folders: Mapping[str, _Folder],
) -> dict[str, _Folder]:
    if front.kind == "decision" and front.reads:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare reads:; a decision agent reads no files, so give it what it needs in its input or its questions"
        )
    names = [_written(agent, name) for name in front.reads]
    for name in names:
        if name not in folders:
            choices = ", ".join(sorted(folders)) or "none"
            raise ConfigurationError(
                f"tree: agent {agent!r} reads unknown folder {name!r}; reads: names a folder by its path from the root, one of: {choices}"
            )
    for name in names:
        if not folders[name].place.is_relative_to(root.resolve()):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {name!r}, which resolves outside the root to {folders[name].place}; reads: needs a folder resolved inside the root, so drop the link or move the folder in"
            )
    for name in names:
        if path == name or path.startswith(f"{name}/"):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {name!r}, its own folder or one that holds it; reads: names other folders, so drop it"
            )
        if path == "." or name.startswith(f"{path}/"):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {name!r}, inside its own folder; its map holds that folder already, so drop it"
            )
    for one, other in combinations(names, 2):
        if one == other or one.startswith(f"{other}/") or other.startswith(f"{one}/"):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {one!r} and {other!r}, one inside the other; keep one"
            )
    if names and directory.resolve() != root.resolve() / path:
        raise ConfigurationError(
            f"tree: agent {agent!r} declares reads: but its folder is a link to {directory.resolve()}; reads: needs the agent's folder inside the root, so link the whole root instead"
        )
    return {name: folders[name] for name in names}


def _under(
    reads: Mapping[str, _Folder], files: Callable[[_Folder], list[tuple[str, Path]]]
) -> list[tuple[str, Path]]:
    return [
        (f"{name}/{rel}", path) for name, folder in reads.items() for rel, path in files(folder)
    ]


def load_agent(
    directory: Path,
    media: Mapping[str, Collection[str] | None] | None = None,
    folders: Mapping[str, _Folder] | None = None,
    path: str = ".",
    tree_root: Path | None = None,
) -> AgentSpec:
    front, body = parse_frontmatter_doc(
        _read(directory / "AGENT.md"),
        model=AgentFront,
        source=directory / "AGENT.md",
        label="agent",
    )
    agent = front.name
    questions = _questions(directory / "questions")
    tools = _tools(directory, agent)
    task_files = _markdown_files(directory / "tasks")
    task_questions: dict[str, dict[str, Question]] = {}
    task_bodies: dict[str, str] = {}
    if front.kind == "decision":
        if task_files:
            first = task_files[0].stem
            raise ConfigurationError(
                f"tree: decision agent {agent!r} has tasks/{first}.md, but keeps each task's questions in a folder; move it to tasks/{first}/"
            )
        task_questions = {p.name: _questions(p) for p in _folders(directory / "tasks")}
        tasks = dict.fromkeys(task_questions, "")
    else:
        loaded = {p.stem: _task(p) for p in task_files}
        tasks = {name: said for name, (said, _) in loaded.items()}
        task_bodies = {name: body for name, (_, body) in loaded.items()}
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
    held = _reads(agent, front, directory, tree_root or directory, path, folders or {})
    reads = from_agent(path, held)
    root = directory.resolve()
    files = sorted([*content(root, False, inner=True), *_under(reads, lambda folder: folder.files)])
    try:
        preload, read = (
            preload_parts(
                sorted(
                    rel
                    for rel, _ in [
                        *content(root, True),
                        *_under(reads, lambda folder: folder.preloaded),
                    ]
                ),
                (media or {}).get(front.model),
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
        path=path,
        kind=front.kind,
        model=front.model,
        description=front.description,
        body=body.strip(),
        agents=front.agents,
        route=front.route,
        questions=questions,
        tools=tools,
        tasks=tasks,
        task_bodies=task_bodies,
        fingerprint=folder_fingerprint(files, read),
        workspace=front.workspace,
        task_questions=task_questions,
        preload=preload,
        preload_digest=preload_digest(preload),
        listing=joined(
            entries(root, True), {name: folder.mapped for name, folder in reads.items()}
        ),
        reads={name: folder.place for name, folder in held.items()},
        skills={} if front.kind == "decision" else _skills(content(root, True), agent),
    )


def load_tree(
    root: Path, media: Mapping[str, Collection[str] | None] | None = None
) -> dict[str, AgentSpec]:
    root = Path(root)
    found, links = _tree_folders(root)
    _refuse_links(root, found, links)
    paths = [path for path in [".", *found] if declares(root / path)]
    if not paths:
        raise ConfigurationError(
            f"tree: {root} holds no AGENT.md; give each agent a folder with an AGENT.md that names its name:, kind: and model:"
        )
    folders = {name: _Folder(place) for name, place in found.items()}
    agents: dict[str, AgentSpec] = {}
    for path in paths:
        spec = load_agent(root / path, media, folders, path=path, tree_root=root)
        twin = agents.get(spec.name)
        if twin is not None:
            first, second = (root / twin.path / "AGENT.md", root / spec.path / "AGENT.md")
            raise ConfigurationError(
                f"tree: {first} and {second} both name the agent {spec.name!r}; rename one"
            )
        agents[spec.name] = spec
    return dict(sorted(agents.items()))


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
