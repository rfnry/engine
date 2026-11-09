import json
import logging
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
from rfnry_engine.folders import (
    DECLARED,
    content,
    declares,
    entries,
    joined,
    leads_out,
    unwarned,
    warns_once,
)
from rfnry_engine.frontmatter import (
    body_of,
    description,
    parse_frontmatter_doc,
    plainly,
)
from rfnry_engine.jail import Jail
from rfnry_engine.models import Check, Choice, Part, Question, Score
from rfnry_engine.tools.builtins import BUILTINS, preload_parts
from rfnry_engine.tools.delegate import DELEGATE_TOOL
from rfnry_engine.tools.folder import (
    NOWHERE,
    ToolDef,
    ToolFolder,
    load_scripts,
    load_tools,
    project_of,
)
from rfnry_engine.tools.history import HISTORY_TOOL
from rfnry_engine.tools.workspace import WORKSPACE_NAMES
from rfnry_engine.workers.pool import Script

logger = logging.getLogger(__name__)

Kind = Literal["generation", "decision"]
OUTPUT_TOOL = "Output"
HANDED_DOWN = frozenset({"skills", "tools"})


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
    to: dict[str, str]
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("to", mode="before")
    @classmethod
    def _an_entry_names_no_task(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        for option, target in value.items():
            if isinstance(target, dict) and "task" in target:
                raise ValueError(
                    f"tree: tasks are gone (route entry {option!r} names task {target['task']!r}); name the agent alone, and make the task a skill or say it in the agent's AGENT.md"
                )
        return value


def _each_skill_once(value: list[str], wanted: str) -> list[str]:
    for entry in value:
        if not entry.strip() or entry != entry.strip():
            raise ValueError(
                f"tree: an entry must not be blank or padded, got {entry!r}; write the name of {wanted}"
            )
    twice = sorted({entry for entry in value if value.count(entry) > 1})
    if twice:
        raise ValueError(f"tree: {twice[0]!r} is listed twice; list each skill once")
    return value


def _a_word_or_names(value: list[str] | str | None, empty: str) -> list[str] | str:
    if value is None or value == []:
        raise ValueError(f"tree: it is empty; {empty}")
    if isinstance(value, str) and value not in ("all", "none"):
        raise ValueError(f"tree: write all, none or a list of names, got {value!r}")
    return value


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
    tools: list[str] | str | None = None
    skills: list[str] | str | None = None
    public_skills: list[str] | str | None = None

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

    @field_validator("tools", "skills", "public_skills", mode="before")
    @classmethod
    def _a_scalar_is_a_word(cls, value: object) -> object:
        if value is not None and not isinstance(value, (list, str)):
            raise ValueError(f"tree: write all, none or a list of names, got {value!r}")
        return value

    @field_validator("tools")
    @classmethod
    def _tools_keep_each_once(cls, value: list[str] | str | None) -> list[str] | None:
        value = _a_word_or_names(
            value,
            "write tools: all to keep every tool, tools: none to keep none, or the tools to keep, such as tools: [lookup_order]",
        )
        if isinstance(value, str):
            return None if value == "all" else []
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

    @field_validator("skills")
    @classmethod
    def _skills_keep_each_once(cls, value: list[str] | str | None) -> list[str] | None:
        value = _a_word_or_names(
            value,
            "write skills: all to keep every skill, skills: none to keep none, or the skills to keep, such as skills: [escalate]",
        )
        if isinstance(value, str):
            return None if value == "all" else []
        return _each_skill_once(value, "a skill to keep, such as skills: [escalate]")

    @field_validator("public_skills")
    @classmethod
    def _public_skills_can_be_typed(cls, value: list[str] | str | None) -> list[str] | str | None:
        value = _a_word_or_names(
            value,
            "write public_skills: none so no one can call a skill, public_skills: all so people may call every skill the agent keeps, or the skills people may call, such as public_skills: [escalate]",
        )
        if isinstance(value, str):
            return None if value == "none" else value
        _each_skill_once(value, "a skill people may call, such as public_skills: [escalate]")
        for entry in value:
            if len(entry.split()) > 1:
                raise ValueError(
                    f"tree: {entry!r} holds whitespace, so no one can type /{entry}; rename the skill to one word, such as my-skill"
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
    fingerprint: str
    workspace: bool = False
    scripted: dict[str, tuple[dict[str, JsonValue], Script]] = Field(default_factory=dict)
    preload: list[Part] = Field(default_factory=list)
    preload_digest: str
    listing: str = ""
    reads: dict[str, Path] = Field(default_factory=dict)
    skills: dict[str, Skill] = Field(default_factory=dict)
    public_skills: tuple[str, ...] = ()
    skill_folders: dict[str, Path] = Field(default_factory=dict)
    folder_tools: dict[str, ToolDef] = Field(default_factory=dict)


def _held(path: Path, agent: Path) -> None:
    target = leads_out(path, agent)
    if target is not None:
        raise ConfigurationError(
            f"tree: {path} links to {target}, outside the agent's folder; move the file into the agent's folder, or link the agent's whole folder"
        )


def _own(agent: Path) -> None:
    for name in sorted(DECLARED):
        target = leads_out(agent / name, agent)
        if target is not None:
            raise ConfigurationError(
                f"tree: {agent / name} links to {target}, outside the agent's folder; move the folder into the agent's folder, or link the agent's whole folder"
            )
        target = (agent / name).resolve()
        parts = target.relative_to(agent.resolve()).parts
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            raise ConfigurationError(
                f"tree: {agent / name} links to {target}, a hidden folder the agent's fingerprint leaves out; move the folder to a visible name"
            )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"tree: {path} is not UTF-8 text; save it as UTF-8") from exc
    except OSError as exc:
        raise ConfigurationError(
            f"tree: cannot read {path}: {exc.strerror or exc}; make it readable to the engine"
        ) from exc


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


def _no_tasks(agent: Path) -> None:
    folder = agent / "tasks"
    if not folder.is_dir():
        return
    try:
        listed = sorted(folder.iterdir())
    except OSError:
        listed = []
    found = [p for p in listed if p.suffix == ".md" and not p.name.startswith(".") and p.is_file()]
    raise ConfigurationError(
        f"tree: tasks are gone ({found[0] if found else folder}); make it a skill, or say it in the input"
    )


def _declared(folder: Path) -> list[Path]:
    try:
        with os.scandir(folder) as entries:
            found = []
            for entry in entries:
                try:
                    entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                found.append(Path(entry.path))
    except OSError as exc:
        raise ConfigurationError(
            f"tree: cannot read {folder}: {exc.strerror or exc}; make it readable to the engine"
        ) from exc
    return sorted(found)


def _skills_in(folder: Path, holder: Path, shown: str) -> dict[str, Skill]:
    if folder.is_symlink() or not folder.is_dir():
        return {}
    found: dict[str, Skill] = {}
    for entry in _declared(folder):
        if entry.name.startswith(("_", ".")) or entry.name in ("AGENT.md", "SKILL.md"):
            continue
        if entry.suffix == ".md" and entry.is_file():
            name, rel, path = entry.stem, entry.name, entry
        elif entry.is_dir() and not entry.is_symlink() and not declares(entry):
            name, rel, path = entry.name, f"{entry.name}/SKILL.md", entry / "SKILL.md"
            if path not in _declared(entry) or not path.is_file():
                continue
        else:
            continue
        if leads_out(path, holder) is not None:
            continue
        if name in found:
            raise ConfigurationError(
                f"tree: {folder} has the skill {name!r} as both {name}.md and {name}/SKILL.md; keep one"
            )
        raw = _read(path)
        found[name] = Skill(
            path=f"{shown}/{rel}", description=description(raw, path) or "", body=body_of(raw)
        )
    return dict(sorted(found.items()))


def _question_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        p
        for p in _declared(directory)
        if p.is_file() and not p.name.startswith(("_", ".")) and p.suffix in QUESTION_FILES
    ]


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
    for path in definitions.values():
        _held(path, directory.parent)
    fixed = {name: _question(path) for name, path in definitions.items() if name not in scripts}
    fronts = {name: _front(definitions[name]) for name in scripts}
    folders = {*script_folders, directory.resolve()}
    loaded = load_scripts(directory, scripts, project_of(root), root, folders) if scripts else {}
    return fixed, {name: (fronts[name], loaded[name]) for name in scripts}


class _Folder:
    def __init__(self, place: Path, apart: Collection[Path] = ()) -> None:
        self.place = place
        self.apart = apart

    @cached_property
    def files(self) -> list[tuple[str, Path]]:
        return content(self.place, False, inner=True)

    @cached_property
    def preloaded(self) -> list[tuple[str, Path]]:
        return content(self.place, declares(self.place), apart=self.apart)

    @cached_property
    def mapped(self) -> list[str]:
        return entries(self.place, declares(self.place), self.apart)


def from_agent[T](path: str, reads: Mapping[str, T]) -> dict[str, T]:
    return {posixpath.relpath(name, path): place for name, place in reads.items()}


def _is_folder(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def _subfolders(folder: Path) -> list[Path]:
    try:
        with os.scandir(folder) as entries:
            return sorted(Path(entry.path) for entry in entries if _is_folder(entry))
    except OSError as exc:
        if unwarned(folder):
            logger.warning("tree: cannot read %s: %s; it is left out", folder, exc)
        return []


def _utf8(name: str, path: Path) -> None:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        raise ConfigurationError(
            f"tree: the name of {str(path)!r} is not UTF-8; rename it"
        ) from None


def _loops(top: Path, found: Mapping[str, Path], key: str, target: Path) -> bool:
    parts = key.split("/")
    above = [top, *(found["/".join(parts[:n])] for n in range(1, len(parts)))]
    return any(place.is_relative_to(target) for place in above)


def _is_agent(folder: Path, path: str) -> bool:
    return not HANDED_DOWN.intersection(path.split("/")) and declares(folder)


def _tree_folders(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    top = root.resolve()
    found: dict[str, Path] = {}
    links: dict[str, Path] = {}

    def walk(folder: Path, at: str) -> None:
        holds = declares(folder)
        if _is_agent(folder, at):
            _no_tasks(folder)
        for child in _subfolders(folder):
            if (
                child.name.startswith(".")
                or child.name == "__pycache__"
                or (holds and child.name in DECLARED)
            ):
                continue
            _utf8(child.name, child)
            place = child.resolve()
            key = f"{at}/{child.name}" if at else child.name
            if child.is_symlink():
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
        if not child.name.startswith(".")
        and child.name not in HANDED_DOWN
        and not child.is_symlink()
    )


def _refuse_links(root: Path, found: Mapping[str, Path], links: Mapping[str, Path]) -> None:
    top = root.resolve()
    reached: dict[Path, list[str]] = {}
    for other, place in found.items():
        reached.setdefault(place, []).append(other)
    for key, target in sorted(links.items()):
        inside = target.is_relative_to(top)
        shown = root / target.relative_to(top) if inside else target
        if _loops(top, found, key, target):
            raise ConfigurationError(
                f"tree: {root / key} links to {shown}, a folder on the way to the link or one holding it; remove the link, since the tree would lead into itself"
            )
        if inside and _holds_agent(target):
            raise ConfigurationError(
                f"tree: {root / key} links to {shown}, a folder of the root that holds an AGENT.md; remove the link, since an agent lives in one folder and is called by its name: from anywhere"
            )
        twice = sorted(other for other in reached.get(target, []) if other != key)
        if not inside and twice and _holds_agent(target):
            parts = twice[0].split("/")
            other = next(
                at for n in range(len(parts), 0, -1) if (at := "/".join(parts[:n])) in links
            )
            raise ConfigurationError(
                f"tree: {root / key} and {root / other} both reach {shown}, which holds an AGENT.md; remove one of the links, since an agent lives in one folder and is called by its name: from anywhere"
            )


def _from_agent(agent: str, verb: str, written: str, path: str) -> str:
    plain = written.removesuffix("/")
    segments = plain.split("/")
    ups = 0
    if segments[0] == ".":
        rest = segments[1:]
    else:
        while ups < len(segments) and segments[ups] == "..":
            ups += 1
        rest = segments[ups:]
    if (
        not plain
        or "\\" in plain
        or (segments[0] != "." and not ups)
        or any(not s or s.startswith(".") for s in rest)
    ):
        raise ConfigurationError(
            f"tree: agent {agent!r} {verb} {written!r}, which is not a path from its folder; write it as './knowledge' or '../shared', with no leading /, no .. after a name, no hidden folder, no empty segment and no backslash"
        )
    parts = [] if path == "." else path.split("/")
    if ups > len(parts):
        raise ConfigurationError(
            f"tree: agent {agent!r} {verb} {written!r}, above the root of the tree; a path in AGENT.md stays inside the tree"
        )
    return "/".join([*parts[: len(parts) - ups], *rest]) or "."


def _spelled(names: Collection[str], path: str) -> list[str]:
    shown = [posixpath.relpath(name, path) for name in names]
    return [one if one == "." or one.startswith("..") else f"./{one}" for one in shown]


def _holds(folder: str, path: str) -> bool:
    return path == folder or path.startswith(f"{folder}/")


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
    skills: Collection[str],
) -> dict[str, _Folder]:
    if front.kind == "decision" and front.reads:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare reads:; a decision agent reads no files, so give it what it needs in its input or its questions"
        )
    entries = [(_from_agent(agent, "reads", said, path), said) for said in front.reads]
    names = [name for name, _ in entries]
    for name, said in entries:
        if _within(name, tools):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {said!r}, a tools/ folder or a folder inside one; a tools/ folder gives its tools to the agents below it and is not read as content, so drop it from reads:"
            )
    for name, said in entries:
        if _within(name, skills):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {said!r}, a skills/ folder or a folder inside one; a skills/ folder gives its skills to the agents below it and is not read as content, so drop it from reads:"
            )
    for name, said in entries:
        if name != "." and name not in folders:
            offered = (
                folder
                for folder in folders
                if not _within(folder, [*tools, *skills]) and not _holds(folder, path)
            )
            choices = ", ".join(_spelled(sorted(offered), path)) or "none"
            raise ConfigurationError(
                f"tree: agent {agent!r} reads unknown folder {said!r}; reads: names a folder by its path from the agent's folder, one of: {choices}"
            )
    for name, said in entries:
        if name != "." and not folders[name].place.is_relative_to(root.resolve()):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {said!r}, which resolves outside the root to {folders[name].place}; reads: needs a folder resolved inside the root, so drop the link or move the folder in"
            )
    for name, said in entries:
        if name == "." or path == name or path.startswith(f"{name}/"):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {said!r}, its own folder or one that holds it; reads: names other folders, so drop it"
            )
        if path == "." or name.startswith(f"{path}/"):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {said!r}, inside its own folder; its map holds that folder already, so drop it"
            )
    for (one, one_said), (other, other_said) in combinations(entries, 2):
        if one == other or one.startswith(f"{other}/") or other.startswith(f"{one}/"):
            raise ConfigurationError(
                f"tree: agent {agent!r} reads {one_said!r} and {other_said!r}, one inside the other; keep one"
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


def _skill_places(root: Path, prefixes: Collection[str]) -> frozenset[Path]:
    return frozenset((root / prefix / "skills").resolve() for prefix in prefixes)


def _gives_skills(folder: Path) -> bool:
    return folder.is_dir() and not folder.is_symlink()


def _skills_from(tree: Path, prefix: str, path: str) -> dict[str, Skill]:
    place = (tree / prefix).resolve()
    folder = place / "skills"
    return _skills_in(
        folder, place if prefix == path else folder, posixpath.relpath(f"{prefix}/skills", path)
    )


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


def _kept(agent: str, front: AgentFront) -> list[str] | None:
    if front.tools is None:
        return None
    declared = {
        HISTORY_TOOL,
        *([DELEGATE_TOOL] if front.agents else []),
        *(WORKSPACE_NAMES if front.workspace else []),
    }
    for name in front.tools:
        if name in declared:
            raise ConfigurationError(
                f"tree: agent {agent!r} keeps {name!r} in tools:, which comes with its own declaration and stays whatever tools: lists; leave it out of tools:"
            )
    return list(front.tools)


def _agent_front(directory: Path) -> tuple[AgentFront, str]:
    _held(directory / "AGENT.md", directory)
    return parse_frontmatter_doc(
        _read(directory / "AGENT.md"),
        model=AgentFront,
        source=directory / "AGENT.md",
        label="agent",
    )


def load_agent(
    directory: Path,
    folders: Mapping[str, _Folder] | None = None,
    path: str = ".",
    tree_root: Path | None = None,
    tool_folders: Mapping[str, dict[str, ToolDef]] | None = None,
    script_folders: Collection[Path] = (),
    skill_prefixes: Collection[str] = (),
) -> AgentSpec:
    front, body = _agent_front(directory)
    agent = front.name
    if front.kind == "generation" and _question_files(directory / "questions"):
        raise ConfigurationError(
            f"tree: generation agent {agent!r} cannot declare questions/; questions are for decision agents, so make it kind: decision or move them to one"
        )
    questions, scripted = _questions(
        directory / "questions", tree_root or directory, script_folders
    )
    root = directory.resolve()
    tree = tree_root or directory
    gives = [] if front.kind == "decision" else [p for p in _on_path(path) if p in skill_prefixes]
    merged = {
        name: skill for prefix in gives for name, skill in _skills_from(tree, prefix, path).items()
    }
    if front.kind == "decision" and "skills" in front.model_fields_set:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare skills:; skills are for generation agents"
        )
    if front.skills is not None:
        for name in front.skills:
            if name not in merged:
                have = (
                    f"its skills are {', '.join(sorted(merged))}" if merged else "it has no skills"
                )
                raise ConfigurationError(
                    f"tree: {directory}/AGENT.md keeps {name!r} in skills:, a skill agent {agent!r} does not have; {have}"
                )
    skills = dict(
        sorted((n, s) for n, s in merged.items() if front.skills is None or n in front.skills)
    )
    if front.kind == "decision" and "public_skills" in front.model_fields_set:
        raise ConfigurationError(
            f"tree: decision agent {agent!r} cannot declare public_skills:; skills are for generation agents"
        )
    if front.public_skills == "all":
        for name in skills:
            if len(name.split()) > 1:
                raise ConfigurationError(
                    f"tree: {directory}/AGENT.md makes {name!r} public through public_skills: all, and no one can type /{name}; rename the skill to one word, or list the skills people may call"
                )
    listed = [] if front.public_skills == "all" else front.public_skills or []
    for name in listed:
        if name not in skills:
            have = f"its skills are {', '.join(skills)}" if skills else "it has no skills"
            raise ConfigurationError(
                f"tree: {directory}/AGENT.md names {name!r} in public_skills:, a skill agent {agent!r} does not have; {have}"
            )
    if front.kind == "decision" and "tools" in front.model_fields_set:
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
    skill_paths = [posixpath.normpath(f"{prefix}/skills") for prefix in skill_prefixes]
    held = _reads(agent, front, directory, tree, path, folders or {}, declared, skill_paths)
    reads = from_agent(path, held)
    sees = [] if front.kind == "decision" else [p for p in _on_path(path) if p in tool_folders]
    apart = _tool_places(tree, tool_folders) | _skill_places(tree, skill_prefixes)
    above = [
        (posixpath.relpath(f"{prefix}/{handed}/{rel}", path), file)
        for handed, seen in (("tools", sees), ("skills", gives))
        for prefix in seen
        if prefix != path
        for rel, file in content(tree / prefix / handed, False)
    ]
    files = sorted(
        [*content(root, False, inner=True), *_under(reads, lambda folder: folder.files), *above]
    )
    for rel, file in files:
        _utf8(rel, file)
    try:
        preload, read = (
            preload_parts(
                sorted(
                    rel
                    for rel, _ in [
                        *content(root, True, apart=apart),
                        *_under(reads, lambda folder: folder.preloaded),
                    ]
                ),
                Jail(root, also=[folder.place for folder in reads.values()]),
                front.tools is None or "Load" in front.tools,
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
        keep=_kept(agent, front),
        fingerprint=folder_fingerprint(files, read),
        workspace=front.workspace,
        scripted=scripted,
        preload=preload,
        preload_digest=preload_digest(preload),
        listing=joined(
            entries(root, True, apart), {name: folder.mapped for name, folder in reads.items()}
        ),
        reads={name: folder.place for name, folder in held.items()},
        skills=skills,
        public_skills=tuple(
            sorted(skills if front.public_skills == "all" else front.public_skills or ())
        ),
        skill_folders={
            posixpath.normpath(f"{prefix}/skills"): (tree / prefix / "skills").resolve()
            for prefix in gives
            if prefix != path
        },
        folder_tools=folder_tools,
    )


@warns_once()
def load_tree(root: Path) -> dict[str, AgentSpec]:
    root = Path(root)
    found, links = _tree_folders(root)
    _refuse_links(root, found, links)
    paths = [path for path in [".", *found] if _is_agent(root / path, path)]
    if not paths:
        raise ConfigurationError(
            f"tree: {root} holds no AGENT.md; give each agent a folder with an AGENT.md that names its name:, kind: and model:"
        )
    for path in paths:
        _own(root / path)
    for path in paths:
        if _agent_front(root / path)[0].kind == "generation":
            _skills_from(root, path, path)
    project = project_of(root)
    on_paths = list(dict.fromkeys(p for path in paths for p in _on_path(path)))
    prefixes = [prefix for prefix in on_paths if (root / prefix / "tools").is_dir()]
    skill_prefixes = [prefix for prefix in on_paths if _gives_skills(root / prefix / "skills")]
    asking = [root / path / "questions" for path in paths]
    script_folders = frozenset(
        place.resolve()
        for place in [*(root / prefix / "tools" for prefix in prefixes), *asking]
        if place.is_dir()
    )
    for prefix in prefixes:
        _declared(root / prefix / "tools")
    loaded: dict[str, ToolFolder] = {}
    for prefix in prefixes:
        above = next(
            (loaded[held].reach for held in reversed(_on_path(prefix)[:-1]) if held in loaded),
            NOWHERE,
        )
        loaded[prefix] = load_tools(root / prefix / "tools", project, root, script_folders, above)
    tool_folders = {prefix: folder.tools for prefix, folder in loaded.items()}
    apart = _tool_places(root, tool_folders) | _skill_places(root, skill_prefixes)
    folders = {name: _Folder(place, apart) for name, place in found.items()}
    agents: dict[str, AgentSpec] = {}
    for path in paths:
        spec = load_agent(
            root / path,
            folders,
            path=path,
            tree_root=root,
            tool_folders=tool_folders,
            script_folders=script_folders,
            skill_prefixes=skill_prefixes,
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
            routed = agents.get(target)
            if routed is None:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} points at unknown agent {target!r}; the tree has {', '.join(sorted(agents))}"
                )
            if routed.kind == "decision" and not routed.questions and not routed.scripted:
                raise ConfigurationError(
                    f"tree: route of {spec.name!r} gives decision agent {target!r} nothing to ask: it has no questions/; give it questions/, since a route hands it only the input"
                )
