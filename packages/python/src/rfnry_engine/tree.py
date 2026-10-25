import json
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
from rfnry_engine.tools.builtins import BUILTINS, preload_parts
from rfnry_engine.tools.delegate import DELEGATE_TOOL
from rfnry_engine.tools.folder import ToolDef, load_scripts, load_tools, project_of
from rfnry_engine.tools.history import HISTORY_TOOL
from rfnry_engine.tools.workspace import WORKSPACE_NAMES
from rfnry_engine.workers.pool import Script

Kind = Literal["generation", "decision"]
OUTPUT_TOOL = "Output"


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
    tools: list[str] | None = None

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

    @field_validator("tools")
    @classmethod
    def _tools_keep_each_once(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            raise ValueError(
                "tree: it is empty; write tools: [] to keep none of the folder and file tools, or leave tools: out to keep them all"
            )
        for entry in value:
            if not entry.strip() or entry != entry.strip():
                raise ValueError(
                    f"tree: an entry must not be blank or padded, got {entry!r}; write the name of a tool to keep, such as tools: [lookup_order]"
                )
        twice = sorted({entry for entry in value if value.count(entry) > 1})
        if twice:
            raise ValueError(f"tree: {twice[0]!r} is listed twice; list each tool once")
        if OUTPUT_TOOL in value:
            raise ValueError(
                f"tree: {OUTPUT_TOOL!r} cannot be kept by name; it comes with expect= on a run, so leave it out of tools:"
            )
        return value


class QuestionFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "score", "check"]
    instructions: str | dict[str, JsonValue] | list[JsonValue] | None = None
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None] | None = None
    levels: list[str | dict[str, JsonValue] | list[JsonValue]] | None = None
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None

    @field_validator("instructions")
    @classmethod
    def _instructions_say_something(
        cls, value: str | dict[str, JsonValue] | list[JsonValue] | None
    ) -> str | dict[str, JsonValue] | list[JsonValue] | None:
        if value is not None and not (value.strip() if isinstance(value, str) else value):
            raise ValueError("tree: instructions must say something; write the question")
        return value


QUESTION_TYPES: dict[str, type[Choice] | type[Score] | type[Check]] = {
    "choice": Choice,
    "score": Score,
    "check": Check,
}
SHAPE = "type, instructions and its options, levels or yes and no"
TAKES = {"choice": ("options",), "score": ("levels",), "check": ("yes", "no")}
TAKEN = "a choice takes options, a score levels and a check yes and no"
QUESTION_FILES = frozenset({".json", ".py", ".md"})


def foreign(kind: JsonValue, data: Mapping[str, JsonValue]) -> str | None:
    own = TAKES.get(kind) if isinstance(kind, str) else None
    if own is None:
        return None
    others = {field for fields in TAKES.values() for field in fields} - set(own)
    return next((key for key in sorted(data) if key in others), None)


def as_question(data: Mapping[str, JsonValue]) -> Question:
    front = QuestionFront.model_validate(data)
    return QUESTION_TYPES[front.type].model_validate(front.model_dump(exclude_unset=True))


class TaskFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None


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
    keep: list[str] | None
    tasks: dict[str, str]
    task_bodies: dict[str, str] = Field(default_factory=dict)
    fingerprint: str
    workspace: bool = False
    scripted: dict[str, tuple[dict[str, JsonValue], Script]] = Field(default_factory=dict)
    preload: list[Part] = Field(default_factory=list)
    preload_digest: str
    listing: str = ""
    reads: dict[str, Path] = Field(default_factory=dict)
    skills: dict[str, Skill] = Field(default_factory=dict)
    folder_tools: dict[str, ToolDef] = Field(default_factory=dict)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"tree: {path} is not UTF-8 text; save it as UTF-8") from exc


def _object(path: Path) -> dict[str, JsonValue]:
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"tree: {path} is not valid JSON: {exc}; fix it in that file"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"tree: {path} is not a JSON object; a question is an object with {SHAPE}"
        )
    unknown = sorted(key for key in data if key not in QuestionFront.model_fields)
    if unknown:
        raise ConfigurationError(
            f"tree: {path} has the unknown key {unknown[0]!r}; a question takes type, instructions, options, levels, yes and no"
        )
    held = foreign(data.get("type"), data)
    if held is not None:
        raise ConfigurationError(
            f"tree: {path} is a {data['type']} question holding {held!r}; {TAKEN}, so drop {held!r} or change its type"
        )
    return data


def _question(path: Path) -> Question:
    try:
        return as_question(_object(path))
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}; {plainly(exc)}") from exc


def _front(path: Path) -> dict[str, JsonValue]:
    try:
        return QuestionFront.model_validate(_object(path)).model_dump(exclude_unset=True)
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}; {plainly(exc)}") from exc


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


def _question_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(("_", ".")) and p.suffix in QUESTION_FILES
    )


def _questions(
    directory: Path, root: Path, script_folders: Collection[Path]
) -> tuple[dict[str, Question], dict[str, tuple[dict[str, JsonValue], Script]]]:
    files = _question_files(directory)
    if not files:
        return {}, {}
    for path in files:
        if path.suffix == ".md":
            raise ConfigurationError(
                f"tree: question {path} is markdown; write it as {path.stem}.json with {SHAPE}"
            )
    definitions = {p.stem: p for p in files if p.suffix == ".json"}
    scripts = [p.stem for p in files if p.suffix == ".py"]
    for name in scripts:
        if name not in definitions:
            raise ConfigurationError(
                f"tree: {directory / name}.py has no {name}.json beside it; a question is {name}.json, and {name}.py fills it from data"
            )
    fixed = {name: _question(path) for name, path in definitions.items() if name not in scripts}
    fronts = {name: _front(definitions[name]) for name in scripts}
    folders = {*script_folders, directory.resolve()}
    loaded = load_scripts(directory, scripts, project_of(root), root, folders) if scripts else {}
    return fixed, {name: (fronts[name], loaded[name]) for name in scripts}


class _Folder:
    def __init__(self, place: Path, tools: Collection[Path] = ()) -> None:
        self.place = place
        self.tools = tools

    @cached_property
    def files(self) -> list[tuple[str, Path]]:
        return content(self.place, False, inner=True)

    @cached_property
    def preloaded(self) -> list[tuple[str, Path]]:
        return content(self.place, declares(self.place), tools=self.tools)

    @cached_property
    def mapped(self) -> list[str]:
        return entries(self.place, declares(self.place), self.tools)


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
            if (
                child.name.startswith(".")
                or child.name == "__pycache__"
                or (holds and child.name in DECLARED)
            ):
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


def tree_folders(root: Path) -> list[str]:
    return list(_tree_folders(root)[0])


def _holds_agent(folder: Path) -> bool:
    return declares(folder) or any(
        _holds_agent(child)
        for child in _subfolders(folder)
        if not child.name.startswith(".") and child.name != "tools" and not _linked(child)
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


def _within(name: str, folders: Collection[str]) -> bool:
    return any(name == folder or name.startswith(f"{folder}/") for folder in folders)


def _reads(
    agent: str,
    front: AgentFront,
    directory: Path,
    root: Path,
    path: str,
    folders: Mapping[str, _Folder],
    tools: Collection[str],
) -> dict[str, _Folder]:
    if front.kind == "decision" and front.reads:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare reads:; a decision agent reads no files, so give it what it needs in its input or its questions"
        )
    names = [_written(agent, name) for name in front.reads]
    for name in names:
        if _within(name, tools):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {name!r}, a tools/ folder or a folder inside one; a tools/ folder gives its tools to the agents below it and is not read as content, so drop it from reads:"
            )
    for name in names:
        if name not in folders:
            offered = (folder for folder in folders if not _within(folder, tools))
            choices = ", ".join(sorted(offered)) or "none"
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


def _tool_places(root: Path, prefixes: Collection[str]) -> frozenset[Path]:
    return frozenset((root / prefix / "tools").resolve() for prefix in prefixes)


def _on_path(path: str) -> list[str]:
    parts = [] if path == "." else path.split("/")
    return [".", *("/".join(parts[:n]) for n in range(1, len(parts) + 1))]


def _folder_tools(
    agent: str, front: AgentFront, seen: list[dict[str, ToolDef]]
) -> dict[str, ToolDef]:
    tools = dict(sorted({name: tool for held in seen for name, tool in held.items()}.items()))
    given = {
        *BUILTINS,
        HISTORY_TOOL,
        OUTPUT_TOOL,
        *([DELEGATE_TOOL] if front.agents else []),
        *(WORKSPACE_NAMES if front.workspace else []),
    }
    for name, tool in tools.items():
        if name in given:
            raise ConfigurationError(
                f"tree: {tool.folder / name}.json is named like {name!r}, a tool the engine gives agent {agent!r}; rename the tool"
            )
    return tools


def _kept(agent: str, front: AgentFront, tools: Collection[str]) -> list[str] | None:
    if front.tools is None:
        return None
    seen = sorted({*tools, *BUILTINS})
    for name in front.tools:
        if name not in seen:
            raise ConfigurationError(
                f"tree: agent {agent!r} keeps {name!r} in tools:, which it does not see; its tools are {', '.join(seen)}"
            )
    return list(front.tools)


def load_agent(
    directory: Path,
    media: Mapping[str, Collection[str] | None] | None = None,
    folders: Mapping[str, _Folder] | None = None,
    path: str = ".",
    tree_root: Path | None = None,
    tool_folders: Mapping[str, dict[str, ToolDef]] | None = None,
    script_folders: Collection[Path] = (),
) -> AgentSpec:
    front, body = parse_frontmatter_doc(
        _read(directory / "AGENT.md"),
        model=AgentFront,
        source=directory / "AGENT.md",
        label="agent",
    )
    agent = front.name
    if front.kind == "decision" and (directory / "tasks").is_dir():
        raise ConfigurationError(
            f"tree: decision agent {agent!r} has tasks/; a decision agent asks only questions/, so make another decision agent for another set of questions"
        )
    if front.kind == "generation" and _question_files(directory / "questions"):
        raise ConfigurationError(
            f"tree: generation agent {agent!r} cannot declare questions/; questions are for decision agents, so make it kind: decision or move them to one"
        )
    questions, scripted = _questions(
        directory / "questions", tree_root or directory, script_folders
    )
    loaded = {p.stem: _task(p) for p in _markdown_files(directory / "tasks")}
    tasks = {name: said for name, (said, _) in loaded.items()}
    task_bodies = {name: body for name, (_, body) in loaded.items()}
    if front.kind == "decision" and front.tools is not None:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot list tools:; tools are for generation agents, so give them to one it routes to"
        )
    if front.kind == "decision" and front.workspace:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare workspace: true; a workspace is for generation agents"
        )
    if front.kind == "decision" and front.preload:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare preload:; a decision agent reads no files, so give it what it needs in its input or its questions"
        )
    if front.kind == "generation" and front.route is not None:
        raise ConfigurationError(
            f"tree: generation agent {agent!r} cannot declare route:; routes are for decision agents"
        )
    tool_folders = tool_folders or {}
    declared = [posixpath.normpath(f"{prefix}/tools") for prefix in tool_folders]
    held = _reads(agent, front, directory, tree_root or directory, path, folders or {}, declared)
    reads = from_agent(path, held)
    root = directory.resolve()
    sees = [] if front.kind == "decision" else [p for p in _on_path(path) if p in tool_folders]
    tools = _tool_places(tree_root or directory, tool_folders)
    above = [
        (posixpath.relpath(f"{prefix}/tools/{rel}", path), file)
        for prefix in sees
        if prefix != path
        for rel, file in content((tree_root or directory) / prefix / "tools", False)
    ]
    files = sorted(
        [*content(root, False, inner=True), *_under(reads, lambda folder: folder.files), *above]
    )
    try:
        preload, read = (
            preload_parts(
                sorted(
                    rel
                    for rel, _ in [
                        *content(root, True, tools=tools),
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
    folder_tools = _folder_tools(agent, front, [tool_folders[prefix] for prefix in sees])
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
        keep=_kept(agent, front, folder_tools),
        tasks=tasks,
        task_bodies=task_bodies,
        fingerprint=folder_fingerprint(files, read),
        workspace=front.workspace,
        scripted=scripted,
        preload=preload,
        preload_digest=preload_digest(preload),
        listing=joined(
            entries(root, True, tools), {name: folder.mapped for name, folder in reads.items()}
        ),
        reads={name: folder.place for name, folder in held.items()},
        skills={} if front.kind == "decision" else _skills(content(root, True), agent),
        folder_tools=folder_tools,
    )


def load_tree(
    root: Path, media: Mapping[str, Collection[str] | None] | None = None
) -> dict[str, AgentSpec]:
    root = Path(root)
    found, links = _tree_folders(root)
    _refuse_links(root, found, links)
    paths = [
        path for path in [".", *found] if "tools" not in path.split("/") and declares(root / path)
    ]
    if not paths:
        raise ConfigurationError(
            f"tree: {root} holds no AGENT.md; give each agent a folder with an AGENT.md that names its name:, kind: and model:"
        )
    project = project_of(root)
    prefixes = [
        prefix
        for prefix in dict.fromkeys(p for path in paths for p in _on_path(path))
        if (root / prefix / "tools").is_dir()
    ]
    asking = [root / path / "questions" for path in paths]
    script_folders = frozenset(
        place.resolve()
        for place in [*(root / prefix / "tools" for prefix in prefixes), *asking]
        if place.is_dir()
    )
    tool_folders = {
        prefix: load_tools(root / prefix / "tools", project, root, script_folders)
        for prefix in prefixes
    }
    tools = _tool_places(root, tool_folders)
    folders = {name: _Folder(place, tools) for name, place in found.items()}
    agents: dict[str, AgentSpec] = {}
    for path in paths:
        spec = load_agent(
            root / path,
            media,
            folders,
            path=path,
            tree_root=root,
            tool_folders=tool_folders,
            script_folders=script_folders,
        )
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
        by = spec.route.by
        if by in spec.scripted:
            raise ConfigurationError(
                f"tree: route of {spec.name!r} is by {by!r}, whose options come from {by}.py; route by a question whose options are fixed"
            )
        question = spec.questions.get(by)
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
            if routed.kind == "decision" and target.task is not None:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} names task {target.task!r} for decision agent {target.agent!r}, which takes no task; drop task: from that entry"
                )
            if target.task is not None and target.task not in routed.tasks:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} points at agent {target.agent!r}, which has no task {target.task!r}; its tasks are {', '.join(routed.tasks) or 'none'}"
                )
            if routed.kind == "decision" and not routed.questions and not routed.scripted:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} gives decision agent {target.agent!r} nothing to ask: it has no questions/; give it questions/, since a route hands it only the input"
                )
