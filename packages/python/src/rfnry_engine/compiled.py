import json
import os
import re
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from pydantic import JsonValue

from rfnry_engine.chain import preload_digest
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.fingerprint import folder_fingerprint
from rfnry_engine.folders import (
    Gitignored,
    listed,
    mapped,
    pattern_covers,
    pattern_leads_into,
    pattern_reaches,
    readable,
    shown_link,
    warns_once,
)
from rfnry_engine.frontmatter import body_of, lenient_description
from rfnry_engine.jail import Jail
from rfnry_engine.manifest import (
    MANIFEST,
    AgentEntry,
    FilesSection,
    Manifest,
    Resolved,
    SkillsSection,
    ToolEntry,
    overlaid,
    read_manifest,
)
from rfnry_engine.models import Choice, Part, Question
from rfnry_engine.tools.builtins import BUILTINS, preload_parts
from rfnry_engine.tools.delegate import DELEGATE_TOOL
from rfnry_engine.tools.folder import ToolDef, load_tool, project_of
from rfnry_engine.tools.history import HISTORY_TOOL
from rfnry_engine.tools.workspace import WORKSPACE_NAMES
from rfnry_engine.tree import (
    OUTPUT_TOOL,
    QUESTION_FILES,
    AgentSpec,
    Gitignores,
    Skill,
    question_files,
    question_parts,
    read_questions,
    skill_text,
    utf8_name,
)
from rfnry_engine.workers.pool import Script

RUN_DECIDES = {
    OUTPUT_TOOL: "expect= on a run",
    HISTORY_TOOL: "a run's history= and start=",
}
ENGINE_TOOLS = frozenset({*BUILTINS, DELEGATE_TOOL, *WORKSPACE_NAMES, *RUN_DECIDES})
BACKGROUND_JOBS = frozenset({"Shell", "Job", "Jobs"})
SKILL_FILE = "SKILL.md"


class _Asked(NamedTuple):
    questions: dict[str, Question]
    scripted: dict[str, tuple[dict[str, JsonValue], Script]]
    files: dict[str, Path]


class _Reach(NamedTuple):
    on_the_map: dict[str, Path]
    files: dict[str, Path]
    preloaded: list[str]


class _Declared(NamedTuple):
    entries: Mapping[str, ToolEntry]
    tools: Mapping[str, ToolDef]


class _Held(NamedTuple):
    skill: Skill
    raw: bytes


class _Reached(NamedTuple):
    include: dict[str, str]
    preload: dict[str, str]


class _Skills(NamedTuple):
    held: Mapping[str, _Held]
    layers: Mapping[str | None, _Reached]


class _Chosen(NamedTuple):
    offered: frozenset[str]
    excluded: frozenset[str]
    preloaded: list[Skill]
    public: tuple[str, ...]


class _Grant(NamedTuple):
    section: str
    given: bool


class _Picked(NamedTuple):
    names: list[str]
    granted: frozenset[str]


def _who(agent: str, mode: str | None) -> str:
    return f"agent {agent!r}" if mode is None else f"mode {mode!r} of agent {agent!r}"


def _text(root: Path, who: str, what: str, written: str) -> tuple[bytes, str]:
    try:
        raw = (root / written).read_bytes()
    except (OSError, ValueError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ConfigurationError(
            f"compiled: {who} cannot read {what} {written!r}: {reason}; write the path of a readable file from the folder root"
        ) from exc
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(
            f"compiled: {who} reads {written!r}, not UTF-8 text; save the file as UTF-8"
        ) from exc


def _reached_by(root: Path, who: str, written: str) -> list[Path]:
    found = [
        path
        for path in root.glob(written)
        if path.is_file()
        and path.suffix in QUESTION_FILES
        and not path.name.startswith("_")
        and pattern_reaches(written, path.relative_to(root))
    ]
    if not found:
        raise ConfigurationError(
            f"compiled: {who} lists the questions {written!r}, which reach no question file; write a path or a glob from the folder root, such as triage/*.json"
        )
    return found


def _beside(directory: Path, reached: Collection[Path]) -> list[Path]:
    stems = {path.stem for path in reached}
    return [
        path
        for path in question_files(directory)
        if path.suffix != ".md"
        and (
            path.stem in stems or (path.suffix == ".py" and not path.with_suffix(".json").is_file())
        )
    ]


def _questions(root: Path, who: str, written: list[str]) -> _Asked:
    folders: dict[Path, set[Path]] = {}
    for pattern in written:
        for path in _reached_by(root, who, pattern):
            folders.setdefault(path.parent, set()).add(path)
    questions: dict[str, Question] = {}
    scripted: dict[str, tuple[dict[str, JsonValue], Script]] = {}
    held: dict[str, Path] = {}
    reached: list[Path] = []
    for directory, paths in sorted(folders.items()):
        definitions, scripts = question_parts(
            directory, sorted({*paths, *_beside(directory, paths)})
        )
        for name, path in definitions.items():
            if name in held:
                raise ConfigurationError(
                    f"compiled: {who} asks {name!r} twice, in {held[name].relative_to(root)} and {path.relative_to(root)}; a question is named by its file, so rename one"
                )
            held[name] = path
        fixed, filled = read_questions(directory, definitions, scripts, root, ())
        questions |= fixed
        scripted |= filled
        reached.extend(definitions.values())
        for _, script in filled.values():
            reached.extend([script.path, *(helper.path for helper in script.helpers.values())])
    return _Asked(
        questions, scripted, {path.relative_to(root).as_posix(): path for path in reached}
    )


def _check_what_it_reaches(root: Path, who: str, files: Mapping[str, Path]) -> None:
    manifest = (root / MANIFEST).resolve()
    for name, path in files.items():
        utf8_name(name, path)
        target = path.resolve()
        if target == manifest:
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which is {MANIFEST}; a model never reads that file, so point at another"
            )
        if not target.is_relative_to(root):
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which links to {target}, outside the folder root; move the file into the folder"
            )


def _in_reach(root: Path, patterns: Collection[str], gitignore: bool) -> dict[str, Path]:
    manifest = (root / MANIFEST).resolve()
    gitignores = Gitignores()
    gitignored = Gitignored(root) if gitignore else None
    found: dict[str, Path] = {}

    def ignored(path: Path, target: Path) -> bool:
        if gitignored is None:
            return False
        if target != path:
            for place in target.parents:
                if place.is_relative_to(root):
                    gitignores.held_in(place, gitignored)
        return gitignored(path)

    def visit(folder: Path, rel: PurePosixPath, trail: frozenset[Path]) -> None:
        if gitignored is not None:
            gitignores.held_in(folder, gitignored)
        for name in listed(folder):
            path, below = folder / name, rel / name
            target = Path(os.path.realpath(path))
            if (
                name == "__pycache__"
                or target == manifest
                or target in trail
                or not target.is_relative_to(root)
                or ignored(path, target)
            ):
                continue
            if path.is_dir():
                if shown_link(root, path) and any(
                    pattern_leads_into(pattern, below) for pattern in patterns
                ):
                    visit(path, below, trail | {target})
            elif (
                any(pattern_covers(pattern, below) for pattern in patterns)
                and path.is_file()
                and readable(path)
            ):
                found[below.as_posix()] = path

    visit(root, PurePosixPath("."), frozenset({root}))
    return found


def _matching(
    who: str, key: str, patterns: Iterable[str], candidates: Mapping[str, Path]
) -> dict[str, Path]:
    matched: dict[str, Path] = {}
    for pattern in patterns:
        if PurePosixPath(pattern) == PurePosixPath(MANIFEST):
            raise ConfigurationError(
                f"compiled: {who} lists {pattern!r} in {key}, which is {MANIFEST}; a model never reads that file, so point at another"
            )
        found = {
            name: path
            for name, path in candidates.items()
            if pattern_covers(pattern, PurePosixPath(name))
        }
        if not found:
            raise ConfigurationError(
                f"compiled: {who} lists {pattern!r} in {key}, which reaches nothing; write a folder, a file or a glob from the folder root that reaches a file, or remove it"
            )
        matched |= found
    return matched


def _own_files(resolved: Resolved, also: Collection[str]) -> set[str]:
    if resolved.instructions is None:
        return set(also)
    return {PurePosixPath(resolved.instructions).as_posix(), *also}


def _reach(
    root: Path,
    who: str,
    files: FilesSection,
    own: Collection[str],
    in_skills: Mapping[str, Path],
) -> _Reach:
    include = files.include or []
    preload = files.preload or []
    candidates = _in_reach(root, [*include, *preload], files.gitignore)
    included = _matching(who, "files.include", include, candidates)
    offered = _matching(who, "files.preload", preload, candidates)
    excluded = _matching(
        who, "files.exclude", files.exclude or [], {**included, **offered, **in_skills}
    )
    left = {name: path for name, path in candidates.items() if name not in excluded}
    kept = {name: path for name, path in included.items() if name in left}
    preloaded = _matching(who, "files.preload", preload, left)
    return _Reach(
        on_the_map={name: path for name, path in kept.items() if name not in own},
        files={**kept, **preloaded},
        preloaded=list(preloaded),
    )


def _skill_folders(root: Path, pattern: str, gitignore: bool) -> list[str]:
    return [
        file.parent.as_posix()
        for file in map(PurePosixPath, _in_reach(root, [pattern], gitignore))
        if file.name == SKILL_FILE and file.parent.name and pattern_reaches(pattern, file.parent)
    ]


def _hold_once(who: str, held: dict[str, str], folder: str) -> None:
    name = PurePosixPath(folder).name
    if held.setdefault(name, folder) != folder:
        raise ConfigurationError(
            f"compiled: {who} reaches two skills named {name!r}, {held[name]} and {folder}; a skill is named by its folder, so rename one"
        )


def _folders_reached(
    root: Path, who: str, key: str, patterns: Iterable[str], gitignore: bool
) -> dict[str, str]:
    held: dict[str, str] = {}
    for pattern in patterns:
        folders = _skill_folders(root, pattern, gitignore)
        if not folders:
            raise ConfigurationError(
                f"compiled: {who} lists {pattern!r} in {key}, which reaches no skill; a skill is a folder holding {SKILL_FILE}, so write such a folder or a glob from the folder root that reaches one, or remove it"
            )
        for folder in folders:
            _hold_once(who, held, folder)
    return held


def _reached_skills(root: Path, who: str, resolved: Resolved) -> _Reached:
    section, gitignore = resolved.skills, resolved.files.gitignore
    return _Reached(
        _folders_reached(root, who, "skills.include", section.include or (), gitignore),
        _folders_reached(root, who, "skills.preload", section.preload or (), gitignore),
    )


def _held_skill(root: Path, who: str, folder: str) -> _Held:
    written = f"{folder}/{SKILL_FILE}"
    utf8_name(written, root / written)
    raw, text = _text(root, who, "the skill", written)
    return _Held(
        Skill(
            path=written,
            place=root / folder,
            description=lenient_description(text, root / written) or "",
            body=body_of(text),
        ),
        raw,
    )


def _every_skill(root: Path, agent: str, entry: AgentEntry) -> _Skills:
    who = _who(agent, None)
    layers = {
        mode: _reached_skills(root, _who(agent, mode), resolved)
        for mode in (None, *entry.modes)
        if (resolved := overlaid(entry, mode)).instructions is not None
    }
    folders: dict[str, str] = {}
    for reached in layers.values():
        for folder in (*reached.include.values(), *reached.preload.values()):
            _hold_once(who, folders, folder)
    return _Skills(
        {name: _held_skill(root, who, folders[name]) for name in sorted(folders)}, layers
    )


def _skill_folder(held: _Held) -> PurePosixPath:
    return PurePosixPath(held.skill.path).parent


def _excluded_skills(who: str, written: Iterable[str], held: Mapping[str, _Held]) -> set[str]:
    excluded: set[str] = set()
    for entry in written:
        found = {
            name
            for name, one in held.items()
            if entry == name or pattern_reaches(entry, _skill_folder(one))
        }
        if not found:
            raise ConfigurationError(
                f"compiled: {who} lists {entry!r} in skills.exclude, which reaches no skill it names; write a skill's name, its folder or a glob from the folder root, or remove it"
            )
        excluded |= found
    return excluded


def _preload_left(
    who: str, patterns: Iterable[str], reached: Mapping[str, str], left: Collection[str]
) -> None:
    for pattern in patterns:
        if not any(pattern_reaches(pattern, PurePosixPath(reached[name])) for name in left):
            raise ConfigurationError(
                f"compiled: {who} lists {pattern!r} in skills.preload, which skills.exclude takes back out; take it out of one of the two"
            )


def _public_skills(who: str, section: SkillsSection, offered: Collection[str]) -> tuple[str, ...]:
    names = sorted(offered if section.public == "all" else section.public or ())
    for name in names:
        if any(character.isspace() for character in name):
            raise ConfigurationError(
                f"compiled: {who} makes {name!r} public, and no one can type it with a space in it; rename the skill's folder to one word, or list only the skills people may call"
            )
        if name not in offered:
            have = f"it offers {', '.join(sorted(offered))}" if offered else "it offers no skills"
            raise ConfigurationError(
                f"compiled: {who} lists {name!r} in skills.public, which it does not offer; {have}, so add it to skills.include or remove the name"
            )
    return tuple(names)


def _chosen_skills(
    who: str, section: SkillsSection, held: Mapping[str, _Held], reached: _Reached
) -> _Chosen:
    excluded = _excluded_skills(who, section.exclude or (), held)
    offered = frozenset(reached.include) - excluded
    preloaded = [name for name in reached.preload if name not in excluded]
    _preload_left(who, section.preload or (), reached.preload, preloaded)
    return _Chosen(
        offered,
        frozenset(excluded),
        [held[name].skill for name in preloaded],
        _public_skills(who, section, offered),
    )


def _left_in_skill(
    name: str,
    held: Mapping[str, _Held],
    excluded: Collection[str],
    taken_back: Collection[str],
    found: Mapping[str, Path],
) -> dict[str, Path]:
    folder = _skill_folder(held[name])
    inside = [
        _skill_folder(held[other])
        for other in excluded
        if folder in _skill_folder(held[other]).parents
    ]
    written = held[name].skill.path
    return {
        written: held[name].skill.place / SKILL_FILE,
        **{
            path: place
            for path, place in found.items()
            if not any(pattern_covers(pattern, PurePosixPath(path)) for pattern in taken_back)
            and not any(other in PurePosixPath(path).parents for other in inside)
        },
    }


def _together(groups: Iterable[Mapping[str, Path]]) -> dict[str, Path]:
    return {name: path for group in groups for name, path in group.items()}


def _with_folders(root: Path, files: Iterable[Path]) -> set[Path]:
    held = {root}
    for file in files:
        held.add(file)
        held.update(parent for parent in file.parents if parent.is_relative_to(root))
    return held


def _outside(root: Path, reached: Mapping[str, Path]) -> Callable[[Path], bool]:
    spelled = _with_folders(root, (root / name for name in reached))
    real = _with_folders(root, (path.resolve() for path in reached.values()))

    def withheld(path: Path) -> bool:
        return (
            Path(os.path.normpath(path)) not in spelled or Path(os.path.realpath(path)) not in real
        )

    return withheld


def _hidden_names(root: Path, reached: Mapping[str, Path]) -> frozenset[Path]:
    held: set[Path] = set()
    for name, path in reached.items():
        for parts in (PurePosixPath(name).parts, path.resolve().relative_to(root).parts):
            held.update(
                root.joinpath(*parts[: depth + 1])
                for depth, part in enumerate(parts)
                if part.startswith(".")
            )
    return frozenset(held)


def _grants(resolved: Resolved) -> dict[str, _Grant]:
    opens_files = any(
        section.include is not None or section.preload is not None
        for section in (resolved.files, resolved.skills)
    )
    return {
        **dict.fromkeys(BUILTINS, _Grant("files or skills", opens_files)),
        DELEGATE_TOOL: _Grant(
            "orchestration.delegate_to", bool(resolved.orchestration.delegate_to)
        ),
        **dict.fromkeys(WORKSPACE_NAMES, _Grant("workspace: true", bool(resolved.workspace))),
    }


def _matching_names(pattern: str, names: Iterable[str]) -> list[str]:
    shape = re.compile(".*".join(map(re.escape, pattern.split("*"))))
    return [name for name in names if shape.fullmatch(name)]


def _names(who: str, key: str, written: list[str] | None, declared: Collection[str]) -> list[str]:
    names: list[str] = []
    for entry in written or ():
        if entry in RUN_DECIDES:
            raise ConfigurationError(
                f"compiled: {who} lists {entry!r} in tools.{key}; the run decides that tool through {RUN_DECIDES[entry]}, so leave it out"
            )
        if "*" not in entry:
            names.append(entry)
            continue
        found = _matching_names(entry, declared)
        if not found:
            raise ConfigurationError(
                f"compiled: {who} lists the pattern {entry!r} in tools.{key}, which matches no name under tools:; write one that matches a declared tool, or remove it"
            )
        names.extend(found)
    return names


def _check_sections(who: str, included: Iterable[str], grants: Mapping[str, _Grant]) -> None:
    for name in included:
        if name in grants and not grants[name].given:
            raise ConfigurationError(
                f"compiled: {who} lists {name!r} in tools.include, which comes with {grants[name].section}, and it has none; give it that section or remove the name"
            )


def _check_jobs_go_together(who: str, dropped: Collection[str]) -> None:
    named = BACKGROUND_JOBS.intersection(dropped)
    if named and named != BACKGROUND_JOBS:
        raise ConfigurationError(
            f"compiled: {who} lists only {', '.join(sorted(named))} of Shell, Job and Jobs in tools.exclude; Shell can start a job only Job and Jobs can read or stop, so exclude the three together or none of them"
        )


def _picked_tools(who: str, resolved: Resolved, declared: Collection[str]) -> _Picked:
    grants = _grants(resolved)
    dropped = set(_names(who, "exclude", resolved.tools.exclude, declared))
    included = _names(who, "include", resolved.tools.include, declared)
    if resolved.instructions is not None:
        _check_sections(who, included, grants)
        _check_jobs_go_together(who, dropped)
    return _Picked(
        [name for name in dict.fromkeys(included) if name not in dropped],
        frozenset(name for name, grant in grants.items() if grant.given) - dropped,
    )


def _tool_files(root: Path, tool: ToolDef, entry: ToolEntry) -> dict[str, Path]:
    paths = [tool.script, *(held.path for held in tool.helpers.values())]
    if entry.schema_file is not None:
        paths.append(root / entry.schema_file)
    return {path.relative_to(root).as_posix(): path for path in paths}


def _preloaded(
    root: Path, who: str, names: list[str], named: frozenset[Path], loadable: bool
) -> tuple[list[Part], dict[str, bytes]]:
    try:
        return preload_parts(names, Jail(root, named=named), loadable)
    except InputError as exc:
        raise ConfigurationError(
            f"compiled: {who} cannot preload: {exc}; fix that file, or drop it from files.preload"
        ) from exc


def _fingerprint(
    root: Path,
    agent: str,
    mode: str | None,
    resolved: Resolved,
    tools: Mapping[str, ToolEntry],
    files: Mapping[str, Path],
    read: Mapping[str, bytes],
) -> str:
    entry = f"{MANIFEST}#{agent}" if mode is None else f"{MANIFEST}#{agent}:{mode}"
    canonical = json.dumps(
        {
            "agent": resolved.model_dump(mode="json"),
            "tools": {name: tool.model_dump(mode="json") for name, tool in tools.items()},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return folder_fingerprint(
        sorted([(entry, root / MANIFEST), *files.items()]), {**read, entry: canonical}
    )


def _spec(
    root: Path,
    agent: str,
    entry: AgentEntry,
    mode: str | None,
    declared: _Declared,
    skills: _Skills,
) -> AgentSpec:
    resolved = overlaid(entry, mode)
    who = _who(agent, mode)
    asked = _questions(root, who, resolved.questions or [])
    picked = _picked_tools(who, resolved, declared.entries)
    folder_tools = {name: declared.tools[name] for name in picked.names if name in declared.tools}
    tool_files: dict[str, Path] = {}
    for name, tool in folder_tools.items():
        tool_files |= _tool_files(root, tool, declared.entries[name])
    files = {**asked.files, **tool_files}
    read: dict[str, bytes] = {}
    body = ""
    said = resolved.description
    if resolved.instructions is not None:
        read[resolved.instructions], text = _text(
            root, who, "its instructions", resolved.instructions
        )
        files[resolved.instructions] = root / resolved.instructions
        body = body_of(text)
        said = said or lenient_description(text, root / resolved.instructions)
    reached = skills.layers.get(mode)
    chosen = (
        _Chosen(frozenset(), frozenset(), [], ())
        if reached is None
        else _chosen_skills(who, resolved.skills, skills.held, reached)
    )
    in_folders = {
        name: _in_reach(root, [_skill_folder(one).as_posix()], resolved.files.gitignore)
        for name, one in skills.held.items()
    }
    skill_files = {
        name: _left_in_skill(
            name, skills.held, chosen.excluded, resolved.files.exclude or (), found
        )
        for name, found in in_folders.items()
    }
    offered_files = _together(skill_files[name] for name in chosen.offered)
    every_skill_file = _together(skill_files.values())
    reach = (
        _Reach({}, {}, [])
        if resolved.instructions is None
        else _reach(
            root,
            who,
            resolved.files,
            _own_files(resolved, {*tool_files, *offered_files}),
            _together(in_folders.values()),
        )
    )
    files |= {**reach.files, **every_skill_file}
    _check_what_it_reaches(root, who, files)
    keep = sorted({*picked.granted, *picked.names})
    named = _hidden_names(root, {**reach.files, **every_skill_file})
    preload, loaded = _preloaded(root, who, reach.preloaded, named, "Load" in keep)
    preload += [
        skill_text(skill) for skill in chosen.preloaded if skill.path not in reach.preloaded
    ]
    read |= {one.skill.path: one.raw for one in skills.held.values()} | loaded
    opened = {**reach.files, **offered_files}
    return AgentSpec(
        root=root,
        name=agent,
        path=".",
        kind=resolved.kind,
        model=resolved.model,
        description=said or "",
        body=body,
        agents=(resolved.orchestration.delegate_to or [])
        if DELEGATE_TOOL in picked.granted
        else [],
        route=resolved.orchestration.route,
        questions=asked.questions,
        keep=keep,
        granted=picked.granted,
        fingerprint=_fingerprint(
            root,
            agent,
            mode,
            resolved,
            {name: declared.entries[name] for name in folder_tools},
            files,
            read,
        ),
        workspace=bool(resolved.workspace),
        scripted=asked.scripted,
        preload=preload,
        preload_digest=preload_digest(preload),
        skills={name: one.skill for name, one in skills.held.items()},
        offered=chosen.offered,
        public_skills=chosen.public,
        folder_tools=dict(sorted(folder_tools.items())),
        listing="\n".join(mapped(sorted(reach.on_the_map.items()))),
        needs_mode=resolved.instructions is None,
        worker=resolved.worker or "default",
        reachable=sorted(opened),
        withheld=_outside(root, opened),
        withheld_when_invoked={
            name: _outside(root, {**opened, **skill_files[name]})
            for name in skills.held
            if name not in chosen.offered
        },
        named=named,
        included_folders=[
            pattern for pattern in resolved.files.include or [] if (root / pattern).is_dir()
        ],
        max_model_calls=resolved.max_model_calls,
        tool_timeout=resolved.tools.timeout,
        max_depth=resolved.orchestration.max_depth,
        tool_parallel=resolved.tools.max_parallel,
        delegate_parallel=resolved.orchestration.max_parallel,
        mode=mode,
    )


def _agent(root: Path, name: str, entry: AgentEntry, declared: _Declared) -> AgentSpec:
    skills = _every_skill(root, name, entry)
    spec = _spec(root, name, entry, None, declared, skills)
    modes = {mode: _spec(root, name, entry, mode, declared, skills) for mode in entry.modes}
    return spec.model_copy(update={"modes": modes})


def _refuse_targets(
    agents: Mapping[str, AgentSpec], who: str, spec: AgentSpec, delegates: Iterable[str]
) -> None:
    routed = spec.route.to.values() if spec.route is not None else ()
    for name in (*delegates, *routed):
        target = agents.get(name)
        if target is None:
            raise ConfigurationError(
                f"compiled: {who} names {name!r} in delegate_to or route, and {MANIFEST} holds no such agent; its agents are {', '.join(sorted(agents))}"
            )
        if target.needs_mode:
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which needs a mode; a delegated or routed run takes none, so give {name!r} instructions on its entry or reach another agent"
            )


def _check_route(agents: Mapping[str, AgentSpec], who: str, spec: AgentSpec) -> None:
    route = spec.route
    if route is None or spec.needs_mode:
        return
    if route.by in spec.scripted:
        script = spec.scripted[route.by][1].path.relative_to(spec.root).as_posix()
        raise ConfigurationError(
            f"compiled: {who} routes by {route.by!r}, whose options come from {script}; route by a question whose options are fixed"
        )
    question = spec.questions.get(route.by)
    if not isinstance(question, Choice):
        raise ConfigurationError(
            f"compiled: {who} routes by {route.by!r}, which is not a choice question in its questions:; write by: as the name of a choice question it asks"
        )
    for option, name in route.to.items():
        if option not in question.options:
            raise ConfigurationError(
                f"compiled: {who} routes the option {option!r}, which {route.by!r} does not have; its options are {', '.join(question.options)}"
            )
        target = agents[name]
        if target.kind == "decision" and not target.questions and not target.scripted:
            raise ConfigurationError(
                f"compiled: {who} routes to decision agent {name!r}, which has no questions: to ask; give it questions: on its entry, since a routed run takes only the input and no mode"
            )


def _check_orchestration(
    agents: Mapping[str, AgentSpec], entries: Mapping[str, AgentEntry]
) -> None:
    for agent in agents.values():
        for spec in (agent, *agent.modes.values()):
            who = _who(agent.name, spec.mode)
            delegates = overlaid(entries[agent.name], spec.mode).orchestration.delegate_to
            _refuse_targets(agents, who, spec, delegates or ())
            _check_route(agents, who, spec)


def _refuse_shared_export_names(agents: Mapping[str, AgentSpec]) -> None:
    held = {name: _who(name, None) for name in agents}
    for agent in agents.values():
        for mode in agent.modes:
            exported, who = f"{agent.name}.{mode}", _who(agent.name, mode)
            if exported in held:
                raise ConfigurationError(
                    f"compiled: {held[exported]} and {who} would export under one key, {exported!r}; rename one"
                )
            held[exported] = who


def _refuse_engine_names(tools: Iterable[str]) -> None:
    for name in sorted(tools):
        if name in ENGINE_TOOLS:
            raise ConfigurationError(
                f"compiled: tools.{name} in {MANIFEST} is named like a tool the engine gives; rename it, since the key is the name the model sees"
            )


def _refuse_loose_tools(agents: Mapping[str, AgentSpec], declared: Iterable[str]) -> None:
    picked = {
        name
        for agent in agents.values()
        for spec in (agent, *agent.modes.values())
        for name in spec.folder_tools
    }
    loose = sorted(set(declared) - picked)
    if loose:
        raise ConfigurationError(
            f"compiled: tools.{loose[0]} in {MANIFEST} is picked by nothing; pick it with tools.include on an agent or a mode, or remove it"
        )


@warns_once()
def compile_folder(root: Path, manifest: Manifest) -> dict[str, AgentSpec]:
    _refuse_engine_names(manifest.tools)
    project = project_of(root)
    declared = _Declared(
        manifest.tools,
        {name: load_tool(root, name, entry, project) for name, entry in manifest.tools.items()},
    )
    agents = {
        name: _agent(root, name, manifest.agents[name], declared)
        for name in sorted(manifest.agents)
    }
    _check_orchestration(agents, manifest.agents)
    _refuse_loose_tools(agents, manifest.tools)
    _refuse_shared_export_names(agents)
    return agents


def load_folder(root: Path) -> dict[str, AgentSpec]:
    root = Path(root).resolve()
    return compile_folder(root, read_manifest(root))
