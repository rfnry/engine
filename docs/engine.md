# rfnry-engine

rfnry-engine is an agent harness with built-in dataset generation.

It is the scaffolding that turns a model into an agent — the loop, the tools, delegation between
agents — as a small library: an agent is a folder — its markdown and the tools it calls — running on
a generation model, a decision model, or both. Opt-in, it records every run, gathers labels from
people, the world and models, and exports datasets to evaluate, improve or train what comes next.

An agent is one of two kinds. A **generation agent** puts a language model in a loop with tools
until it has an answer. A **decision agent** asks a decision model a set of typed questions about a
state and gets typed, probabilistic answers back in one call. Agents nest in both directions: a
generation agent can delegate to any agent, a decision agent can route to any agent. Opt-in, a
refinery records what your agents did and exports it as datasets, with every label and its
source — from people, the world and models. What you do with them — evaluate, improve prompts and
rules, train or distill models, audit — is yours.

Every public name, with its signature, is in [the API reference](api.md).

## Philosophy

**The tree is the agent.** Everything a person needs to author, review or diff about an agent is a
file under version control: its markdown, and the definition and the script of each tool it calls.
The library reads the tree; it never writes it.

**The model chooses.** A run gives the model who the agent is and the rules every run follows, a
map of its folder, the list of its skills and the tools it has; the model opens what a question
needs, follows the skill that fits, and calls the tool or the agent that can help. One agent with a
well-kept folder covers many jobs, and whether a second one is worth it is
[measured](#orchestrating), not assumed.

**Two kinds of model, one or both.** Open-ended work — conversation, tool use, writing — is
generation. Closed-set judgement — which of these, how much, is this true — is decision. A
generation model does the first well and the second expensively; a decision model does the second
in milliseconds with calibrated probabilities and cannot do the first at all. The library treats
them as two contracts and lets you build an agent from either or both.

**The library runs; it does not take over.** You bring the models, the conversation history, the
storage connection, the bucket, and the tools that must live in your process. The tools of the tree
run in worker processes the engine starts beside yours, never inside it, until you close the
engine. There is no background loop, no plugin registry, no configuration language. Every verb is a
function you call. It writes nothing of its own: the only writes are a model's, inside a `Workspace`
you pass a run, and the one file the engine itself creates there, the socket a command calls the
run's tools through, which lives as long as that command ([Workspace](#workspace)).

**Nothing is hidden.** A run returns a record: what was sent, what came back, every tool call with
its timing, every model call with its usage, every child run. That record is what you feed your own
telemetry, and it is what the refinery stores and exports — everything that could be useful, for
any purpose, without assuming one.

**Tenancy by construction.** Every run names its scope. Every stored row carries it. There is no
default scope and nothing reads across scopes. Multi-tenancy is the default.

## Getting started

```
agents/
  support/
    AGENT.md                  who the agent is, the rules every run follows, its model
    knowledge/refunds.md      read when a question needs it
    skills/escalate.md        how to do a thing: listed, and opened when it fits
    tools/lookup_order.json   a tool: what the model reads about it
    tools/lookup_order.py     what runs when the model calls it
```

```markdown
---
name: support
kind: generation
model: strong
---
You are the support desk for Acme. Be brief, be exact, cite the rule you are applying. Look an
order up with lookup_order before you answer about it.
```

```json
{
  "description": "Fetch an order by its id. Returns its items, status and charges.",
  "input_schema": {
    "type": "object",
    "properties": {"order_id": {"type": "string", "description": "The order id, such as A-1001"}},
    "required": ["order_id"]
  }
}
```

```python
import json
import os
from urllib.request import urlopen

ORDERS = os.environ["ORDERS_URL"]


def run(args, call):
    with urlopen(f"{ORDERS}/{call['scope']}/orders/{args['order_id']}", timeout=10) as reply:
        return json.load(reply)
```

```python
async with Engine(root=Path("agents"), models={"strong": llm}) as engine:
    run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
    print(run.reply)
```

`llm` is your adapter to a generation model, under the name the agent's `model:` gives it
([Models](#models)). `lookup_order` is the agent's own tool, two files in its `tools/` folder: the
JSON is what the model reads about it — what it does, what it returns and the arguments it takes —
and the script is what runs, `run(args, call)`, in a worker process the engine starts beside your
server, so a tool that crashes takes down its worker, never your server; `call["scope"]` is the
tenant of the run ([Tools](#tools)). `AGENT.md` holds the agent's persona and the rules every run
follows — who it is, what it does, when to use which tool. The model receives `AGENT.md`, its tools,
a map of its folder and the list of its skills, opens `knowledge/refunds.md` when the question needs
it, and answers. One call runs an agent for a tenant and returns its record; leaving `async with`
closes the engine, which stops its workers.

An agent grows by its tree:

| to give an agent | add to the tree | in your code |
| --- | --- | --- |
| the persona and the rules every run follows | `AGENT.md` | |
| a reference, a document, anything else | any file in its folder, in the map with its `description:` | |
| a way of doing something | a [skill](#skills) | |
| a system of yours | a [tool](#tools): `tools/<name>.json` and `<name>.py` | |
| a shell to write and run code | [`workspace: true`](#workspace) | a `Workspace` |
| a fast, calibrated judgement | a [decision agent](#decision-agents), named in `agents:` | a decision model |
| another agent to hand work to | its name in [`agents:`](#nesting) | |
| a memory, or a corpus too large for a folder | | rfnry-memory's `MemoryTools` |

A new agent is a new folder: the code you write grows with the models and systems you connect, not
with the number of agents.

rfnry-engine needs Python 3.14 or later.

## Models

The library speaks to models through two protocols. Any object with the right method is a model;
adapters to vendors are yours to write and never ship here.

```python
class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike`
are protocols over the fields of `Reply`, `Decided` and the answers below, and a result is read by
those fields, not its class. `QuestionLike` is the same over the questions a decision model is
handed; an adapter branches on `question.type` and reads `options`, `levels` or `yes` and `no`. An
adapter typed against another library's copy of these contracts — rfnry-memory's — fits here
unchanged, so one adapter serves both libraries: annotated with `QuestionLike` from either library,
it serves both, statically and at runtime. A result whose fields do not fit is a `ModelError`.

`Message`, `ToolSpec`, `ToolCall` and `Reply` are the message shapes the loop exchanges with a
generation model. An assistant turn is a `Message(role="assistant", content=…, tool_calls=[…])`: its
text in `content`, its calls as `ToolCall`s. A tool result is a `Message(role="tool",
tool_call_id=…, content=…)` whose `is_error` says the call failed, for adapters whose vendor marks
failed tool results. `messages` may hold consecutive `role="user"` messages — the preloaded files
ahead of the history, several messages from the inbox — so an adapter whose vendor needs alternating
roles merges them. A `Reply` whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty
means "run these and call me again"; anything else is final:

```python
Reply(text="", tool_calls=[ToolCall(id="c1", name="Read", args={"path": "knowledge/refunds.md"})], stop_reason="tool_use")
Reply(text="Refunded the second charge.", stop_reason="end_turn", usage=Usage(input=812, output=41), model="…")
```

A message's `content` is a string or a list of parts — one vocabulary for everything a model can
see, shared by every library in the family:

```python
Text(text="Is this invoice charged twice?")
Image(media_type="image/png", data=png)
Document(media_type="application/pdf", name="invoice.pdf", url="https://example.com/invoice.pdf")
Audio(media_type="audio/wav", data=wav)
```

`Part` is any of the four and `Content` is `str | list[Part]`. A media part carries exactly one of
`data` (bytes) or `url`; both or neither is an `InputError`. Bytes travel as base64 in JSON. The
library never fetches a `url` — it carries it to your adapter, which decides; turning parts into a
vendor's blocks, and refusing what the vendor cannot take, is the adapter's job. There is no video
part: frames travel as images. Parts are matched by shape too, so another family library's copy of
them is accepted as it is.

Structured output is a tool call — the model is offered an `Output` tool — so an adapter needs
nothing beyond tool support. An adapter that also implements `generate_stream(system, messages,
tools, on_delta)` streams when the caller asks for events: it awaits `on_delta(text)` with the
reply's text as it arrives and `on_delta(text, reasoning=True)` with the model's reasoning, when its
vendor streams it; one that never passes `reasoning=` streams text alone. An adapter that has
`media_types`, the file types its model takes — `{"image/*", "application/pdf"}` — lets `Load`
refuse the rest with an error the model can act on, instead of the next model call failing; without
it, `Load` hands the model any file. Each agent follows the `media_types` of its own model: `Load`
refuses a file for the agent whose model does not take it, and for that agent alone; nothing about a
file `preload:` sends is refused when the tree loads. An adapter that has `model`, the name of
the model it calls, names the run's model before any reply does. An adapter that has `settings`, a
JSON object of what it sends its vendor besides the conversation — `{"effort": "high",
"max_tokens": 4096}` — has them recorded beside the model; a `Reply` or a `Decided` may carry its
own for one call, and they win, even when empty. An adapter's `settings` attribute that is not a
JSON object is ignored; a reply's or a decided's settings that are not JSON break the contract
(`ModelError`). They are stored and
exported with the run, so a credential or an endpoint never goes in them.

A decision model answers three kinds of question:

```python
Choice(instructions="Which department should handle this?", options={"billing": "…", "technical": "…"})
Score(instructions="How urgent is the request?", levels=["Can wait days", "Today", "Within the hour", "Outage"])
Check(instructions="Does the user ask to speak with a person?")
```

and returns, respectively, a `ChoiceAnswer` (`choice`, `probabilities`), a `ScoreAnswer` (`score`,
`probabilities`), or a `CheckAnswer` (`probability`). Each carries a `confidence` in `[0, 1]` that
the library computes from the distribution's peakedness, `(n·p_max − 1)/(n − 1)` — the same
formula for every adapter and every type, so a threshold means the same thing everywhere. A
generation model can stand in as a decision model through an adapter that asks it for
probabilities; the calibration is worse, the code is the same. `decide` may return the answers
alone, or a `Decided(answers, usage, model, settings)` when the adapter can report what the call cost and
which model answered; the record carries the usage, model and settings the call reported, and the adapter's
`model` and `settings` attributes when the call reports none. A decision adapter may carry
`media_types` too, the media its model reads in a state; without it, its model is handed text.

The instructions, what an option means, each level, and `yes` and `no` are strings, or JSON objects
or lists when a question has parts of its own — the question beside the data it names in
backticks, options told apart by what they cover and what they do not. An option may mean `None`
when its key says enough. The record keeps the structure as it was asked, and the adapter passes it
to its model as it is, or renders it where its model reads a prompt:

```python
Choice(
    instructions={"question": "Which team should handle this message?", "focus": "The main request, not every topic"},
    options={
        "billing": {"what": "Charges, invoices, refunds", "not_for": "Delivery", "examples": ["I was charged twice"]},
        "orders": {"what": "Delivery, cancellation, returns", "examples": ["Where is my package?"]},
        "other": None,
    },
)
```

An agent names its model in `AGENT.md` (`model: strong`), and the engine maps those names to
adapters. The credentials, the endpoint and the vendor's model stay in the adapter you build; the
tree only says which name an agent runs on, so one tree runs on different models in different
environments:

```python
engine = Engine(root, models={"strong": strong_llm, "fast": fast_llm, "judge": dm})
```

A lead on `strong` delegating to a writer on `fast` is two lines in two `AGENT.md` files, and each
runs on its own adapter, in one record that names the model of every run. `models` is required,
and `{}` is allowed for a tree that is only read. Its keys are plain names, like an agent's, and
each value has `generate` or `decide`; anything else is a `ConfigurationError` when the engine is
built. The mapping is checked against the tree then too: a model no agent runs on is refused with
the names the tree uses (`no agent runs on model 'fsat'; the tree's agents run on judge, strong`),
and so is an agent whose kind its model cannot serve (`agent 'writer' is a generation agent, and
model 'judge' has no generate()`). An agent whose model was not given is not an error until it
runs: the others run, and running it, delegating to it or routing to it is a `ConfigurationError`
saying which model it needs (`agent 'writer' runs on model 'fast', which this engine was not
given`), before any model is called. `engine.models` holds the mapping given.

Fakes are enough to run everything in tests:

```python
class Scripted:
    def __init__(self, *replies): self.replies = list(replies)
    async def generate(self, system, messages, tools): return self.replies.pop(0)

class Fixed:
    async def decide(self, state, questions):
        return {"intent": ChoiceAnswer(choice="billing", probabilities={"billing": 0.8, "technical": 0.15, "other": 0.05}),
                "urgent": CheckAnswer(probability=0.91)}
```

## Core features

### An agent is a directory

A folder with an `rfnry.yaml` at its root is read from that file instead of from `AGENT.md` files: one entry per agent under `agents:`, whose `instructions` names any markdown file from the folder root, its frontmatter left out of the prompt, and whose `modes` each compile to a full agent when the engine loads.

`files:` decides what a model may open, and every path in it, in the map and given to the file tools is written from the folder root, so an instructions file saying `knowledge/refunds.md` means that in every call. `include` lists folders, files or globs, `preload` is sent with every model call and stays readable whether or not `include` reaches it, `exclude` takes paths back out of both, so an excluded file is neither sent nor readable, and `gitignore: true` leaves out what a `.gitignore` ignores. The map in the prompt holds what `include` reaches less `exclude` and less the agent's own instructions. A hidden name is reached only where a path in the file names it, as `.claude/agents` does, and hidden names below it stay hidden; the file tools never open `rfnry.yaml`; a link that leads out of the folder is left out, and one that stays inside is read under its own name, a file link to a hidden file included. What is out of reach reads as missing to `Read`, `Load`, `LS`, `Grep` and `Glob`, which come with `files`, and `tools.exclude` drops any of them. Reach is fixed when the engine loads: a file added later reads as missing until a new `Engine` loads the folder. A pattern that reaches nothing is a `ConfigurationError`, so a mistyped path cannot quietly shrink what a model sees.

`skills:` decides which skills a generation agent has. A skill is a folder holding `SKILL.md`, and its name is the folder's name; a flat `<name>.md` is no skill here. `include` and `preload` take a path to such a folder or a glob reaching some, from the folder root, so `.claude/skills/*` loads as it is; a glob's match that holds no `SKILL.md` is left out, and a path or a glob that reaches no skill is a `ConfigurationError`, as is two folders giving one name, naming both. The model is offered `include` less `exclude`, listed in the prompt's `SKILLS` section under the path `Read` takes, with the description its frontmatter gives. `exclude` takes a skill's name, its folder or a glob; it also takes a skill out of `preload`, so an excluded skill is neither listed nor sent, and one that names none of the agent's skills is a `ConfigurationError`. `preload` sends a skill's text, its frontmatter left out, with every model call, in the shape `skill=` sends it, whether or not the list offers it; a `files.preload` of the same `SKILL.md` keeps the file as it is, and the file goes once. `public` holds the names people may type as `/name`, or the word `all` for every offered skill; a name the entry does not offer, or one holding whitespace, is a `ConfigurationError`, since no one could reach it. A run's `skill=` reaches any skill the entry or one of its modes names, offered, preloaded or excluded alike, and a person's `/name` only what `public` names; a delegated or routed run takes no skill. An offered skill's folder leaves the map and stays open to the file tools. A skill not offered is reachable only where `files.include` covers it, but the folder of the skill a run invokes is open for that run. `files.exclude` holds inside a skill's folder, for everything but its `SKILL.md`, which `skills.exclude` decides, and a skill excluded from the list stays closed through a skill it is nested in. `Read`, `Grep`, `Glob`, `LS` and `Load` come with `skills` as they do with `files`, and `gitignore: true` under `files` leaves out of a skill's folder what a `.gitignore` ignores. The frontmatter of an instructions file, a `SKILL.md` or a markdown file on the map is read only for its `description`, so a folder written for another harness loads unchanged: YAML that does not parse is no error, and a top-level `description:` line is still read, its matching quotes stripped. `rfnry.yaml` itself stays strict.

`tools:` beside `agents:` declares each tool once, and an agent's `tools.include` picks it by name; the key is the name the model sees, and neither the file's name nor the function's decides anything, so `CallInfo.tool` and a tool's error text say the key. `module` is a `.py` path from the folder root, because the workers run Python. `function` is what the worker calls as `function(args, call)`, `async def` or plain `def`, `run` when left out, so two tools may share a module, which loads once in a process, even when one tool names it and another imports it; the file is checked to hold it at its top level without being imported. `schema` is optional, the path of a JSON file taking `description`, `input_schema` and `output_schema` and no other key: without one a tool takes any JSON object, checks its own arguments, and its result goes unchecked. `description` comes from the entry, else from the schema file. `timeout` is seconds per call, and the smallest of the tool's, the agent's `tools.timeout` and `Engine(tool_timeout=)` applies; `tools.timeout` bounds the agent's other tools under `Engine(tool_timeout=)` the same way, and a `Workspace`'s own `timeout` still replaces both for `Shell`, `Job`, `Write` and `Edit`. An entry of `tools.include` or `tools.exclude` holding `*`, the only wildcard, is a pattern over the names under `tools:` and never reaches a built-in or a tool a run gives. An entry without one is a name: a declared tool, then a tool the entry's sections grant, then one the run gives through `tools=`, and a run that does not give it fails before any model call. The sections grant `Read`, `Grep`, `Glob`, `LS` and `Load` with `files`, `Delegate` with `orchestration.delegate_to`, and `Shell`, `Job`, `Jobs`, `Write` and `Edit` with `workspace: true`, whether or not `tools.include` names them. Naming one in `include` when its section is missing is a `ConfigurationError` naming the section, since no run can supply it. `Output` and `History` are the run's to decide, through `expect=` and through `history=` with `start=`, so either one in `include` or `exclude` is a `ConfigurationError`. `exclude` applies after `include`, across the entry and its mode, and drops exactly the tool it names, a tool a section grants included: `exclude: [Delegate]` takes delegation away, and `exclude: [Write]` leaves `Edit` and the rest. `Shell`, `Job` and `Jobs` go together or not at all, since `Shell` can start a job only `Job` and `Jobs` can read or stop, so excluding one or two of them is a `ConfigurationError`. A name in `exclude` that is neither declared nor engine-given is read as one the run gives, and that tool of the run is dropped, so a mistyped name there cannot be caught when the engine loads. A run's `tools=` replaces a declared tool of the same name, which is how a test or a replay stubs a real system; a tool of the run that `tools.include` does not name is not offered. A pattern that matches nothing, a declared tool that no entry or mode picks once `exclude` has applied, a declared tool named like one the engine gives (`Read`, `Grep`, `Glob`, `LS`, `Load`, `Shell`, `Job`, `Jobs`, `Write`, `Edit`, `Delegate`, `History`, `Output`) and a module, function or schema that is not there are `ConfigurationError`s when the engine loads.

A mode is a named overlay on one agent, written under its `modes:` and holding only the differences: lists add to the entry's, single values replace them, and `exclude` wins over `include` wherever it was written. A run selects one with `mode=`, `engine.run("frontline", scope, message, mode="chat")`, and the record's `mode` says which. An unknown mode is a `NotFoundError` before any model is called, and an entry with no `instructions` needs a mode, so a run without one is an `InputError` naming its modes. Delegated and routed runs, and a run through `engine.tool`, take no mode: they run on the agent's entry, so `delegate_to` or `route` naming an agent that needs a mode is a `ConfigurationError` when the engine loads. Each mode has its own fingerprint, and a replay runs on the mode the run recorded.

A tool imports the standard library, installed packages and any module beside it, whatever the module is named. A module beside a tool is a `.py` file there: a package folder beside a tool is refused when the engine loads, as is an import of a name below such a file (`import shared.sub`), and installed packages import as usual. A module in a folder above is out of its reach, and an import that resolves to nothing is a `ConfigurationError`, unless it sits under `try/except ImportError` or `if TYPE_CHECKING`. A module beside it is imported at the top of the file, not inside a function, and is not named like a standard-library module. Only what `tools:` names and the modules those bring in by an `import` statement go to a worker and into the agent's fingerprint, along with the tool's entry, so a test file or a `conftest.py` beside a tool is just a file: nothing loads it, no worker receives it and nothing checks its imports. A tool's module runs under a name of its own, so a footer under `if __name__ == "__main__":` stays inert in a worker. The module, the schema and the modules beside it that a tool imports leave the map of an agent that picks it, as its instructions do; the rest of a folder `files.include` reaches stays on the map.

```
agents/
  support/
    AGENT.md              who the agent is and the rules every run follows
    knowledge/refunds.md  any folders and files, all in the map; a description adds a line
    knowledge/rates.pdf   text, images, PDFs, anything — read when needed
    skills/escalate.md    how to do a thing: listed in SKILLS, opened at the model's choice, or invoked
    tools/refund.json     a tool, with refund.py beside it, that the model calls
```

The frontmatter of `AGENT.md` holds these keys, and no others; its body is the persona and the rules
every run follows.

- `name:` names the agent, for every agent.
- `kind:` is `generation` or `decision`, for every agent.
- `model:` names the model the agent runs on, for every agent.
- `description:` is the line other agents see in their `AGENTS` list, for every agent.
- `agents:` lists the agents a generation agent may hand an input to with `Delegate`.
- `route:` sends a decision agent's input to another agent by one of its questions; a decision agent
  only.
- `workspace:` gives a generation agent a shell and files to write in, when `true`.
- `gitignore:` leaves out what a `.gitignore` ignores, when `true`; a generation agent only.
- `preload:` lists the files and folders a generation agent is sent with every input.
- `reads:` lists the other folders a generation agent reads; a generation agent only.
- `tools:` keeps some of the tools a generation agent has, or `all` or `none`.
- `skills:` names the skills a generation agent's model is offered, or `all` or `none`.
- `public_skills:` lists the skills people may call by typing `/name`, or `all` or `none`; a
  generation agent only.

`name:`, `kind:` and `model:` are required: a missing one, a `kind:` other than `generation` or
`decision`, or a blank, padded or non-text `name:` or `model:` is a `ConfigurationError` naming the
file. A `name:` is plain text — not blank, not hidden, not padded, no slash — and is what `run`,
`agents:` and `route:` call the agent by, so a folder can be renamed or moved deeper in the tree
without renaming the agent; two agents with one name are a `ConfigurationError` naming both files.
`model:` names a model, `strong` or `judge` for instance; what the name maps to is the engine's to
say. An agent has the tools of its own `tools/` folder and of the `tools/` folder of every folder
above it, a tool of a folder nearer the agent replacing one of the same name above it
([Tools](#tools)):

```markdown
---
name: support
kind: generation
model: strong
description: Handles technical support conversations
agents: [triage]
public_skills: [escalate]
---
You are the support desk for Acme. Be brief, be exact, cite the rule you are applying.
```

The engine does not care where rules live; put the ones every run follows in `AGENT.md`, whose body
is in the prompt of every run. A file the model has to open may not be opened, depending on the
model, so one that must reach every run is named in `preload:`, below.

At the start of a run the model receives `AGENT.md`, a map of its folder, the list of its
[skills](#skills), the list of tools it has, the descriptions of the agents it may call, and any
context the caller passes. The map is generated when the tree loads: every folder and every file,
under its path as the tools take it, and beside a file the `description:` of its frontmatter when it
has one.

The map and the list of skills are shown only to an agent that can open them: one whose tools
include `Read` or `Load`. An agent with neither gets no `FILES` or `SKILLS` section, and a skill
invoked with `skill=` or `/name` still reaches it, since its body goes in the input.

```markdown
---
description: When a refund is owed, and what needs a person
---
A duplicate charge is refunded in full without approval.
```

```
knowledge/
  rates.pdf
  refunds.md — When a refund is owed, and what needs a person
```

A file without a description is listed by its name, which often says enough; a description is one
line of what the file holds, so the model knows what to open without opening it. Other frontmatter
keys are left alone. `AGENT.md` and the folders `questions/`, `skills/` and `tools/` stay out of the
map, since the prompt holds `AGENT.md` and lists the skills and tools, and so do the folders of the
agents inside its own, which it reaches through `agents:` — they stay readable to its tools, as its
declarations do, and count in its fingerprint. Frontmatter that is not valid YAML — a key named
twice included, as YAML forbids, in any file of the map or any skill — or a `description:` that is
not a string, is a `ConfigurationError` naming the file and what to change, and for YAML the line,
when the tree loads, so a typo is caught before any run. Everything else the model reads when it
needs it, with the `Read`, `Grep`, `Glob`, `LS` and `Load` tools, and it can read only inside its
own directory, the folders it reads and the [`skills/` folders above it](#skills), and nothing the
content fingerprint leaves out, which is every name starting with `.` and every `__pycache__`: such
a path is an `InputError` (`tools/builtins: .env is hidden; names starting with . and __pycache__
are not part of the agent's folder`). A path into a `skills/` folder above the agent, as the
`SKILLS` section lists it (`../skills/brief.md`), starts from that folder, so it holds when the
agent's folder, or one above it, is a link. A path is judged where it leads, its links and `..`
resolved, from the agent's folder or the read folder it lies in, as the fingerprint walks them. A
file reads when the folder that holds it has no hidden name on its way there and its own name, as
written, is not hidden: a visible link to a hidden file of the same folder, `notes.md -> .env`,
reads under its own name, the one the fingerprint covers it under, while `kb/../.env` is hidden, and
so is `/elsewhere/git/config` when `/elsewhere/git` links to the agent's `.git`. A folder that `LS`
or `Grep` is given reads when the folder it leads to has no hidden name: a visible link to a hidden
folder, `shown -> .secret`, is hidden, and so is every file read through it, for the fingerprint
never follows a folder link, and `Glob`, `Grep`, `LS` and the map show nothing these rules hide, nor
a link that leads outside those folders, while `LS` lists a broken or looping link inside them by
its name, as it lists a file, and `Grep` given a file labels its matches with the path as given, a
file link under its own name. A part written after a file names a folder that is not there, as POSIX
has it, so `Read`, `Load`, `Grep` and `LS` of `notes.md/`, `notes.md/.`, `notes.md/..` or
`notes.md/x` are `tools/builtins: not a folder: notes.md/`, the path as written. A file link that
comes from outside those folders, or leads from one of them into another, is judged by the file it
leads to as well as by its own name, since the fingerprint covers it only there, and a path that
leads outside them is the jail's error (`jail: path outside the agent directory: <path>`), unless it
leaves them through a hidden name: a link in a hidden folder that leads out, `.secret/notes ->
/elsewhere`, is hidden, and so is every path through it. `Read` returns a text file as text. `Load`
hands the model any other file as it is — an image, a PDF, a recording, a spreadsheet — so a price
list, a diagram or a scanned form sits in the folder as it came, the model reads it with its own
understanding of images and documents, there is no parser in between, and nothing is loaded until
the model asks — unless the agent names it in `preload:`, below.

`AGENT.md` makes a directory an agent, and its frontmatter names it. `Engine(root)` walks `root` and
loads every folder that holds an `AGENT.md` as an agent, at any depth, in folders named and nested
as you choose — `root` itself when it has one, `support/`, `desks/billing/triage/`. A folder without
an `AGENT.md` is a group of agents, or a folder of knowledge the agents read; it is never run.
Hidden folders, an agent's own `questions/`, which stays its declaration, and every `skills/` and
`tools/` folder are never searched for agents, so an `AGENT.md` in them is no agent. An agent inside
another agent's folder is an agent like any other, which the outer one calls through `agents:`; the
root, when it is an agent, is the outer agent of the whole tree. A tree with no `AGENT.md` anywhere
is a `ConfigurationError`.

```
agents/
  lead/
    AGENT.md                   agents: [writer, triage]
    playbook.md                in lead's map
    helpers/notes.md           in lead's map
    helpers/writer/AGENT.md    an agent: out of lead's map, readable to lead's tools
  desks/support/triage/
    AGENT.md                   an agent: desks/ and support/ are groups, never run
```

Folder links are followed: an agent's folder kept elsewhere is linked in and found where the link
is. A link to a folder of the root is not followed — the folder is walked once, under its own path —
and a link that would reach an agent twice is a `ConfigurationError` naming the link and where it
leads: one to a folder of the root that holds an `AGENT.md`, in it or below it, or two links to one
such folder elsewhere, the second link named being the one that leads to the other path; so is a
link to a folder on the way to the link or one holding it, which would lead the walk into itself. A
folder it cannot read is left out, with one warning that names it each time the tree loads, however
many walks meet it: on the `rfnry_engine.tree` logger when the root's walk for agents meets it
first, on `rfnry_engine.folders` when the walk of an agent's folder or one it reads for its map,
preload and fingerprint does. A content file it cannot read is left out the same way, on
`rfnry_engine.folders`: out of the map, the preload and the fingerprint, which is then the
fingerprint of the folder without it. A file the agent is made of — its `AGENT.md`, a skill's
`skills/<name>.md` or `SKILL.md`, a question's `.json`, a tool's `.json` or `.py`, a helper or a
question's script — is never left out: one it cannot read is a `ConfigurationError` naming it
(`tree: cannot read <path>: Permission denied; make it readable to the engine`, or `tools/folder:`
for a script or a tool's file), while any other file of a skill's folder that it cannot read is left
out with the warning. Nor is a folder that declares them left out: a `tools/`, `questions/`,
`skills/` or `skills/<name>/` folder it cannot read or search is a `ConfigurationError` naming that
folder, with no warning before it unless it is, or lies in, a `tools/` or `skills/` folder of a
folder without an `AGENT.md`, which the root's walk for agents meets first and warns about — a
decision agent has no skills, so its `skills/` is left out with the warning. A skill is read from a
`skills/` folder itself, the agent's own or one [above it](#skills): a `<name>.md` in it but an
`AGENT.md`, or the `SKILL.md` of a `<name>/` folder in it that holds no `AGENT.md`. A skill file
that is a link counts when it leads to a file in the agent's folder — for a `skills/` folder above
it, in that `skills/` folder — a hidden one included, and is never read when it leads out of it,
while no skill is read through a folder link under `skills/`, or a `skills/` that is itself a link,
one inside the agent's folder or any above it. Nothing behind one in the agent's own is checked;
behind one above it the walk for agents goes on, as through any folder link. A file removed while
the engine lists its folder — an editor's swap file, an rsync temporary, a ConfigMap's `..data` swap
— is left out as though it was never there; one that cannot be read for any other reason is refused
as above. A file or folder whose name is not UTF-8 is a `ConfigurationError` naming it: `tree: the
name of '/srv/agents/support/caf\udce9.md' is not UTF-8; rename it`. What the engine loads stays
where the content fingerprint finds it, since the fingerprint leaves out a link that leads out of
the folder it walks: a tool's `.json`, `.py` or helper that is a link to a place outside its
`tools/` folder, which the fingerprint walks on its own for every agent that gets its tools, is a
`ConfigurationError` (`tools/folder: <path> links to <target>, outside its folder; move the file
into its folder, or link the agent's whole folder`, or `the whole tools/ folder` for the `tools/`
folder of a folder without an `AGENT.md`), and so is a question's `.json`, `.py` or helper or an
`AGENT.md` that is a link to a place outside the agent's folder, and an agent's own `tools/`,
`questions/` or `skills/` folder that is a link to a folder outside the agent's (`tree: <path> links
to <target>, outside the agent's folder; move the folder into the agent's folder, or link the
agent's whole folder`) or to a hidden folder inside it, which the fingerprint leaves out as it
leaves out every hidden name (`tree: <path> links to <target>, a hidden folder the agent's
fingerprint leaves out; move the folder to a visible name`). Any other link that stays inside loads,
and so do an agent's whole folder linked in and a `tools/` folder of a folder without an `AGENT.md`
linked from elsewhere, which the agents below it walk where it leads.

A root holds agents and folders without an `AGENT.md` side by side, and an agent reads other folders
of its root by naming them in `reads:` — folders without an `AGENT.md`, or other agents' folders:

```
my_agents/
  coding/AGENT.md       reads: ['../books']
  writing/AGENT.md      reads: ['../books']
  books/                no AGENT.md: read by both, never run
```

```markdown
---
name: coding
kind: generation
model: strong
description: Writes and reviews code
reads: ['../books']
---
You write code. The books hold the house's conventions.
```

Each folder it reads follows its own in the map as `../books/`, and the tools take its files by that
path — `Read` of `../books/python/async.md`, `Grep` with `path="../books"`, `Glob` of
`../books/**/*.md`; without a path, `Grep` and `Glob` search the agent's own folder. Nothing outside
the agent's folder, the folders it reads and the [`skills/` folders above it](#skills) is reachable.
Another agent's folder shows in the map without its `AGENT.md` and declarations, which stay
readable, and a folder read shows without the agents inside it or the `skills/` and `tools/` folders
on their paths, as an agent's own map does. `preload:` may name a folder it reads, or a file in one,
under the same paths, and an edit to a folder read changes the fingerprint of every agent that reads
it. `reads:` names a folder by its path from the agent's own folder, another agent's or not, written
with `/` and a trailing `/` allowed and quoted in the list: `'../books'` for an agent directly under
the root, `'../../shared/books'` for one two folders deep. It is the spelling the map and the tools
show, and the one every path in `AGENT.md` takes. `reads:` names folders only, never a file. Each of
these is a `ConfigurationError` when the tree loads: a path that does not start with `./` or `../`,
or that holds `\`, an empty segment, a hidden folder or a `..` after a name; a path above the root
of the tree; a folder the tree does not have; a [`tools/`](#tools) or [`skills/`](#skills) folder on
an agent's path, or a folder inside one, which gives its tools or skills and is not read as content;
a link that leaves the root; the agent's own folder or one that holds it; a folder inside the
agent's own, which its map holds already, so every `reads:` of a root agent; two folders that are
the same or one inside the other; `reads:` on a decision agent, which reads no files; and `reads:`
on an agent whose folder is a link to a folder elsewhere — link the whole root instead. The tree is
shared by every tenant: it holds how to do the work, never a tenant's data — that comes per run, in
the input, the history or the workspace. An agent that reads `my_agents/acme/` reads it for every
tenant.

`engine.agents` lists the tree as the engine loaded it: each name to an `AgentInfo` with its `path`
— its folder from the root, `.` for the root agent — `kind`, `model`, `description`, the `agents` it
may call, the folders it `reads`, the `tools` it gets from its folders, as its `tools:` narrows
them, the `skills` it holds, the ones the code may invoke, and the `public_skills` people may call,
[below](#skills). When the tree has loaded, the engine logs the names
of its agents at INFO on the `rfnry_engine.engine` logger — `engine: agents loaded [lead, triage,
writer]` — so a folder meant to be an agent that is not shows at start;
`logging.basicConfig(level=logging.INFO)` shows it.

An agent with reference files every run needs — a few pages applied to thousands of documents —
names them in `preload:`:

```markdown
---
name: tagger
kind: generation
model: strong
description: Tags each incoming document by its reference files
preload: ['./knowledge']
---
Tag the document. Every reference file applies to every document.
```

`preload:` is the list of what to send ahead: folders and files, each written from the agent's
folder as `reads:` writes one, `'.'` for the agent's own folder; left out, nothing is sent. Every
entry is the agent's folder or inside it, inside a folder in `reads:`, or inside a [`skills/` folder
above it](#skills), so what is sent is what the agent is permitted to reach. A file entry sends that
file, a declared folder's included (`'./skills/x.md'`). A folder entry sends every file below it,
sorted by path character by character, walking as the map does: an agent's `AGENT.md`, `questions/`,
its `skills/` and `tools/` folders and those on an agent's path and the folders of the agents inside
it stay out, and naming one of those folders directly sends it. A skill is preloaded only when it is
named, and stays listed for the model to open the one that fits.

The entries go in the order of the list, each file under the path the map and `Read` use, or, for an
entry that is a link to a folder, under the link's own name, which `Read` takes though the map lists
the files under the folder the link leads to: what `Read` would return for it. A file that is not
text, or is over 5 MB, goes as a line naming its type and size, such as `image/png, 15 bytes; Load
opens it`, which `Load` opens when the model asks; nothing about it is refused when the tree loads.
An agent whose `tools:` leaves out `Load` gets the line without `; Load opens it`, and so does a file
of a skill the model is not offered, since its `Load` could not open it. The map stays a map and the
folder stays as it is; only when the files are read changes, so no turn goes to opening a text file.
They travel as one user message ahead of the history, sent with every model call, and are read when
the engine loads the tree, as the fingerprint is taken, so an edit shows in the next
`Engine`. `run.preload` holds that message's parts; `run.messages` and `run.conversation` leave it
out, so a conversation passed back does not send the folder twice and a replay preloads from the
candidate's folder. The tools stay as they are — `Read` hands over the text the model already holds.
Each agent sends its own list, a delegated or routed one included.

A decision agent reads no files, so `preload:` on one is a `ConfigurationError` when the tree loads.
So are an empty `preload:` or one with no value, one that is not a list, an entry outside the
agent's folder and the folders it reads (`tree: agent 'tagger' preloads '../books/x.md', outside its
folder and the folders it reads; add its folder to reads:, or move the file into the agent's
folder`), or in a folder that holds the agent, which `reads:` cannot name (`tree: agent 'tagger'
preloads '../policy.md', in a folder that holds the agent, and reads: cannot name such a folder;
move the file into the agent's folder or into a folder it reads`), an entry that does not exist
(`tree: agent 'tagger' preloads './nope.md', which does not exist; write its path from the agent's
folder`), an entry whose path cannot be read (`tree: cannot read <path>: Permission denied; make it
readable to the engine`), two entries that name one path or would send one file, a link counted
(`tree: agent 'tagger' preloads './knowledge' and './knowledge/', which send the same file; keep
one`) — a folder entry does not send what it leaves out, so `['.', './skills/x.md']` loads — and a
named file `Read` would refuse or that goes missing or unreadable after the list is read (`tree:
agent 'tagger' cannot preload: tools/builtins: __pycache__/x.pyc is hidden; names starting with .
and __pycache__ are not part of the agent's folder; fix that file, or drop it from preload:`).

`gitignore: true` in an `AGENT.md` leaves out what a `.gitignore` ignores, as Git would, so a folder
kept in a repository can hold its `build/` and its logs without the agent seeing them. In the
agent's own folder, in each folder in `reads:` and in each inherited `skills/` folder, the
`.gitignore` files at or below that folder decide; one above it is not read. Patterns are relative
to their file, a deeper file overrides a shallower one, `!` re-includes, and nothing under an
ignored folder can be re-included. The engine reads each `.gitignore` once, when it loads the tree,
so a change shows in the next `Engine`, and it reads no Git data: a file a pattern matches is
ignored even if Git tracks it.

A file so ignored is not there for that agent. It is left out of the map, of a folder `preload:`
names, of the skills the agent has and of its fingerprint, but never its `AGENT.md` or its own
`tools/`, and `Glob`, `Grep` and `LS` do not list it. `Read`, `Load`, `Grep` or `LS` naming it reads
`tools/builtins: debug.log is ignored by a .gitignore; it is not part of the agent's folder`, and a
`preload:` entry naming it is a `ConfigurationError` through the same message (`tree: agent 'coding'
cannot preload: tools/builtins: debug.log is ignored by a .gitignore; it is not part of the agent's
folder; fix that file, or drop it from preload:`). A link is judged by its own name and by where it
leads. The key is per agent: another agent reading the same folder without `gitignore: true` sees
every file. A tool's script still loads from `tools/` whatever a `.gitignore` says, and `AGENT.md`
and the agent's own `tools/` stay in its fingerprint whatever one says, since the engine runs them.

A workspace sees the real files: a `Shell` command reads whatever the mount holds, ignored or not.
`gitignore: true` is a convenience for what the agent's file tools and prompt hold, not a boundary;
what must stay away from a model stays out of the tree.

A line that is no valid pattern is skipped, as Git skips it, and a `.gitignore` that is a link or
not a plain file is not read. One that cannot be read or decoded is a `ConfigurationError` when the
tree loads (`tree: cannot read agents/coding/.gitignore: Permission denied; make it readable to the
engine`), and so is `gitignore:` on a decision agent (`tree: decision agent 'triage' cannot declare
gitignore:; a decision agent reads no files, so give it what it needs in its input or its
questions`).

### Skills

A skill is how to do something: `skills/<name>.md`, or `skills/<name>/SKILL.md` for one that keeps
files of its own beside it, in a generation agent's own folder or a folder above it; a skill folder
written for another tool — `skills/<name>/SKILL.md` with `name:` and `description:` — works as it
is, its other keys left alone. The prompt lists the skills the model is offered in a section of
their own, after the map: one line per skill, in name order, its path as `Read` takes it and its
`description:` when it has one.

Each audience has its own key. The model is offered what `skills:` names: those skills are listed in
`SKILLS`, as below, and opened with `Read`. The code's `skill=` reaches every skill the agent holds,
offered or not. A person's `/name` reaches what `public_skills:` names. A skill neither offered nor
invoked on this run is not there for the file tools. `preload:` is the other way a skill reaches the
prompt: a skill named there, or a whole `skills/` folder named there, is sent whether or not the
model is offered it.

```
<<<SKILLS>>>
skills/escalate.md — Hand a conversation to a person
```

An agent has the skills of its own `skills/` folder and of the `skills/` folder of every folder
above it — the root's, a group folder's or an outer agent's — as it has the [tools](#tools) of the
`tools/` folders above it: a skill of a folder nearer the agent replaces one of the same name above
it, its own folder's before a group's and a group's before the root's. Each is listed under its path
from the agent's folder, the path `Read` takes:

```
agents/
  skills/brief.md, tone.md     every agent of the tree has them
  desks/
    skills/brief.md            the agents under desks/ have this brief instead of the root's
    support/AGENT.md
    support/skills/close.md    only support, and the agents inside it, have close
```

```
<<<SKILLS>>>
../skills/brief.md — Brief the desk
skills/close.md — Close a ticket
../../skills/tone.md — Keep it warm
```

A `skills/` folder on an agent's path is a declaration, as the agent's own is: it stays out of every
map and of any folder `preload:` names, a folder another agent reads included, and `reads:` cannot
name it; `Read`, `Load`, `Grep`, `Glob` and `LS` reach its files under the path the list shows,
`../skills/plot/run.py` for a file a skill keeps, whether or not a folder on the way is a link, and
a workspace command at `$RFNRY_TREE/desks/skills/plot/run.py`, or at
`$RFNRY_AGENT/../skills/plot/run.py` when no folder on the agent's path is a link, since the shell
follows a link before `..`; and its files count in the fingerprint of every agent that has its
skills. A decision agent has no skills, so a `skills/` folder above it gives it none and leaves its
fingerprint alone. A `skills/` folder above that is a link gives no skills: none is read through it,
and it joins no fingerprint of the agents below it. A skill file in a `skills/` folder above that
links out of that folder is never read. A folder named `skills` off every agent's path is content
like any other, but never holds an agent.

The model opens a skill with `Read` when it judges it fits. The `skills/` folder stays out of the
map and of any folder `preload:` names, so a skill is read only when it is wanted, or sent ahead
when `preload:` names it directly (`'./skills/x.md'`); a skill's other files are found through its
`SKILL.md` and read by their paths under `skills/<name>/`, as any file is, for a skill the model is
offered or the run invoked, a skill neither offered nor invoked being not there, and there is no tool
of its own for skills. A script a skill keeps is a file like any other, which the model reads; to run
it, the agent needs a [workspace](#workspace) whose shell can reach the agent's folder — mount it
read-only, outside the workspace, and say where in `AGENT.md`. A skill's body is read when the tree
loads, so editing the file afterwards changes no prompt; build a new `Engine`.

A run invokes a skill in two ways: from the code with `skill=`, or from a person with an input that
starts with `/<name>`, when the agent makes that skill public:

```python
run = await engine.run("support", scope="acme", input="A third complaint this week.", skill="escalate")
run.skill                                            # "escalate"

run = await engine.run("support", scope="acme", input="/escalate A third complaint.")
run.skill                                            # "escalate"
run.input                                            # "A third complaint."
```

The skill goes in front of the input, in the same user message: a text part of `<<<SKILL
skills/escalate.md>>>` on its own line and the file's body without its frontmatter, then the input
as it came. It is part of the conversation, so a run that continues from it with no `skill=` still
has the skill's text, but not its files unless `skills:` offers it. Delegated and routed runs never
take the caller's skill. `run.skill` records it, an exported generation row carries it, and a replay
asks the candidate for it again, so the candidate's own skill file is the one read.

`engine.agents[name].skills` lists every skill the agent holds, own and inherited, the ones the code
may invoke with `skill=`, offered to the model or not, each to its description, `""` when it has
none. A skill the agent does not hold is a `NotFoundError` before any model call, naming the ones it
holds — `engine: agent 'support' has no skill 'refund'; its skills are escalate` — and `skill=` on a
decision agent is an `InputError`. When the tree loads, one skill as both `skills/x.md` and
`skills/x/SKILL.md` is a `ConfigurationError`. Names that start with `_` or `.` are not skills, and
neither is an `AGENT.md` or a `SKILL.md` directly in `skills/`: a skill file is a `<name>.md` there
or a `<name>/SKILL.md`.

`skills:` in an `AGENT.md` is what the model is offered, as `tools:` is for tools:
`skills: [brief, tone]` offers these two, own or inherited, `skills: none` none, and an agent with
`skills: all`, or without `skills:`, offers every skill it holds. `all` and `none` are words only as
the whole value: inside a list they are names, so `skills: [all]` offers a skill called `all`. An
empty list, a `skills:` with no value, or any other value that is not a list, a word or a number or
`true` included, is a `ConfigurationError` naming the key
(`frontmatter: invalid agent agents/support/AGENT.md; skills: it is empty; write skills: all to offer every
skill, skills: none to offer none, or the skills to offer, such as skills: [escalate]`, or `write all, none or a
list of names, got 'brief'`), and so are `tools:` and `public_skills:` written alike. A skill left
out of `skills:` is not in the `SKILLS` list, but the agent still holds it: the code's `skill=`
reaches it, it stays in `engine.agents[name].skills`, and like every skill held it stays in the
fingerprint. `skills:` that names a skill the agent does not hold is a `ConfigurationError` when
the tree loads — `tree: agents/support/AGENT.md offers 'ghost' in skills:,
a skill agent 'support' does not have; its skills are brief, close` (`it has no skills` when it has
none), or, for a skill a `.gitignore` hides from an agent with `gitignore: true`, `tree:
agents/support/AGENT.md offers 'local' in skills:, a skill a .gitignore hides from agent 'support';
stop ignoring it, or drop it from skills:` — as is a blank or padded entry, a name listed twice, or
`skills:` on a decision agent. `public_skills:` names the same refusal for its entry.

A skill the model is not offered, and that this run did not invoke, is not there for its file tools:
`Read` or `Load` naming it reads exactly as a file that does not exist — `tools/builtins: not a
file: skills/close.md`, or the workspace's wording beside a workspace — `LS` naming it reads as a
folder that does not exist, `Grep` naming it finds nothing, and `LS`, `Glob` and `Grep` leave it out
of what they list, whatever the spelling, `..` and links included, as the `gitignore:` check does.
For `skills/<name>.md` that is the file, for `skills/<name>/SKILL.md` the whole `<name>/` folder,
and a skill of a folder above that a nearer one of the same name replaces follows that name: a file
it shadows is not there when the name is not offered, and stays as any file when it is. A link
elsewhere in the folder that leads to such a skill is left out of the map and of a folder `preload:`
names, judged by where it leads, and stays in the fingerprint. A skill invoked with `skill=` or
`/name` is there on that run, its folder's other files included, and on a replay of that run, while
every other skill left out stays hidden; delegated and routed runs and a run of `engine.tool` invoke
nothing, so they see only what is offered. Only the agent's own and inherited skills count: a
`skills/` folder inside another agent's folder it reads stays as any file of that folder. `preload:`
is unchanged: naming a skill sends it, offered or not. A workspace sees the real files: a `Shell`
command reads whatever the mount holds. This hides a skill from the file tools and the prompt; it is
not a boundary, and what must stay away from a model stays out of the tree.

`public_skills:` in an `AGENT.md` names the skills people may call by typing `/name`. Absent or
`none` it names none, so typing `/name` invokes nothing on an agent that does not list it;
`public_skills: all` makes every skill the model is offered public, so a code-only skill is public
only when named, and an empty list or no value is a
`ConfigurationError` (`public_skills: it is empty; write public_skills: none so no one can call a
skill, public_skills: all so people may call every skill the model is offered, or the skills people may
call, such as public_skills: [escalate]`). `engine.agents[name].public_skills` is the menu a consumer shows people: the public skills
in name order, each to its description, `{}` when there are none. A name in it must be a skill the
agent holds, offered or not — `tree: agents/support/AGENT.md names 'ghost' in public_skills:, a skill
agent 'support' does not have; its skills are brief, close` (`it has no skills` when it has none) —
and, as for `skills:`, a blank or padded entry, a name listed twice, or `public_skills:` on a
decision agent is a `ConfigurationError` when the tree loads, as is a name that holds whitespace,
since no one could type it (`skills:` and `skill=` still take such a name); `public_skills: all`
with an offered skill of that kind is the same error, naming it (`tree: agents/support/AGENT.md
makes 'my skill' public through public_skills: all, and no one can type /my skill; rename the skill
to one word, or list the skills people may call`).

`engine.run` reads an input for a command when it is a string that starts with `/`, at its very
first character, or a list whose first part is a `Text` that does. Its first word is the text up to
the first whitespace; when that word is `/` and the name of a public skill, the run takes that skill
and the input becomes the text after the word, its leading whitespace stripped — in a list the
first part becomes that text, or is dropped when nothing follows, and a list left with no part
becomes `""`. Any other input stays as it came, and is never an error: a path (`/etc/passwd`), an
unknown name, `/escalate/x`, `/escalatex`, ` /escalate`, a JSON object or a JSON list that is not
parts, and every input to a decision agent, which has no skills. A skill the agent has but does not
make public is no command either: the input stays as it came, so typing the name of a code-only
skill invokes nothing and gets the answer any other text would.

`public_skills:` decides what typing invokes, not what the model is offered: `/secret x` reaches the
model as text, and a skill `skills:` offers stays in its `SKILLS` list whether or not it is public,
for an agent that has `Read` or `Load`, where the model may open it with `Read`. A skill the model
should not be offered is left out of `skills:`, which takes it off the list and away from its file
tools; the code still reaches it, and a person only if `public_skills:` names it.

Giving both ways at once is an `InputError` before any model call — `engine: a run invokes one
skill, by skill= or by an input that starts with /<name>; got skill='escalate' and /close` — while
`skill="escalate"` with an input of `/ghost x`, which names no public skill, takes `escalate` and
keeps that input. The run records `skill` and the rest of the input as its `input`, and a replay
sends that pair and never reads `/name` again. `Delegate`, `engine.tool` and routes never read
`/name`: their input comes from a model or a router, and they never take a skill.

### Running an agent

`engine.run(agent, scope, input)` names the agent, the tenant and the input, and returns a record.
For a generation agent that record is a `Generation`: the reply, the system prompt as sent, the
files `preload:` sent (`run.preload`), the messages as sent to the first model call, the whole
conversation, the tools the model was offered (`run.tools`, each a `ToolSpec`), every tool call as a
`Step` with its result and timing, the model calls and their summed usage, and any child runs. It is
the whole observability payload; hand it to your own telemetry:

```python
run.run_id, run.model, run.calls, run.usage.input, run.usage.output, run.duration_ms
for step in run.steps:
    step.tool, step.args, step.is_error, step.duration_ms, step.child_run_id
run.children                                         # runs made through Delegate, in step order
```

`run.model` is the model the replies reported, the resolved one — a dated snapshot, say. When no
reply reported one — a run stopped during its first model call — it is the adapter's `model`
attribute, if it has one that is a non-empty string, and `None` otherwise. `run.settings` is the
settings of the call that wrote the run's last assistant message: that reply's `settings`, else the
adapter's `settings`.

Every run says which agent and which tools produced it, beside the model that answered:

```python
run.fingerprint.agent                                # "3f9a1c0e5b7d2a41": the agent's folder
run.fingerprint.tools                                # "a02c77e1f4b9d305": the tools it was offered
```

`fingerprint.agent` is a short sha256 over every file the agent reads, by relative path and content:
its own directory, the `tools/` folder of each folder above it, whose tools it gets, and the
`skills/` folder of each folder above it, whose skills it holds, offered to the model or not, under
their paths from the agent's folder, `../tools/` and `../skills/`, and each folder it names in
`reads:` with that folder's files under its path from the agent's folder, `../<name>/`
(`../../shared/books/` for an agent two folders deep), as the map shows them — leaving out hidden
files and folders and every file or folder named `__pycache__`, as `LS`, `Glob` and `Grep` do, links
that lead out of them, folder links to a hidden name, and, for an agent with `gitignore: true`, what
a `.gitignore` ignores, but never its `AGENT.md` or its own `tools/`. The engine takes it once, when
it loads the tree, so an edit to the agent, to a folder it reads or to a `skills/` folder above it
shows in the next `Engine` built from the folder.
`fingerprint.tools` is one over `run.tools` — each tool's name, description and schema — so it
covers the tools the agent gets from its folders, the `tools=` a run passes, `Output` when `expect=`
asks for one, in `Delegate` the agents it may call — their names, descriptions and kinds — and a
workspace's tools, whose `Shell` description says whether the workspace has `tree=` and whether its
commands can call tools.
Neither sees what varies per run and is not the agent — the input, the history, `context=`, `tags=`
— so two runs of an unchanged folder share it and any edit changes it, a tool's script included. The
model is `run.model`; the code behind a tool a run passes is yours to version. A decision run
carries one too, its `tools` the fingerprint of no tools and its `questions` the questions it asked
([Decision agents](#decision-agents)), and each delegated or routed run carries its own.

Keep the conversation yourself and pass it back. `run.conversation` is the history the run was
sent, the input, every assistant turn — its text in `content`, its calls in `tool_calls`, its
`reasoning` as the model returned it — every tool result with its `is_error`, and the final
assistant turn, so the next run continues from what the agent did, not only what it said:

```python
run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
next = await engine.run("support", scope="acme", input="And the second charge?", history=run.conversation)
```

A conversation that runs for months is passed whole with where each run starts, and the run sends
only what follows: [Long conversations](#long-conversations).

The input is a string, a JSON object, or a list of parts — what the user sent, as they sent it:

```python
parts = [Text(text="Is this invoice charged twice?"), Image(media_type="image/png", data=png)]
run = await engine.run("support", scope="acme", input=parts)
```

A list of parts becomes the user message after the history as it is; parts in `history` reach the
model unchanged, and a later run continues from them. History, like input, is read by shape: a
message or part from another library is read by its fields.

Pass `record=False` to run without the refinery recording it — a replay, an evaluation, a dry run.
A consult it makes through an `engine.tool`, or a `Delegate` dispatched through an `EngineTools`,
is not recorded either, so a replay whose candidate consults a production engine leaves the scope
as it found it. A run your own tool starts with `engine.run` is recorded unless it passes
`record=False`.

A run that raises instead of returning — the vendor refused the call, the adapter broke the
contract (`ModelError`), the loop passed `max_steps` (`RunawayError`) — returns no record; the
exception is what you have, the same one the adapter or the engine raised. With a refinery
attached, the run is recorded first, as far as it got: `error` says what ended it —
`"ModelError: …"` — `reply` is `""` and `stop_reason` is its last reply's. When `on_event` raises
on one call of a reply, the reply's other calls are still answered, run or cut by a stop, and their
events still reach the handler; the run then raises the first failure and logs each further one as
an error, `engine: a further call of the same batch failed: …`. Its conversation can hold a tool
call with no result, when the run failed between the call and its result, and `run()` refuses such
a history: go on from the history the run was given. A run refused before its first model call — a
bad input, an agent or skill the tree does not have, a route loop — has nothing to record,
and a run whose task you cancel is not recorded; `stop=` ends a run with its record. A generation
run whose `on_event` raises on its `RunStarted` had started: it is recorded with that `error` and
no model call. A delegated run that fails stays under its parent in `children`, its step the error
the parent's model was given, and the parent goes on; a routed run that fails fails its router,
which is recorded with its answers and the failed run in `routed_to`, or `None` when the routed run
was refused before its first model call. A run that failed keeps the delegated runs that had
returned in `children`: those its steps answered, in step order, then those it had not answered yet,
in the order they started, as `refinery.run` reads them back.

Pass `tags=` to mark a run with what you will group it by later — the commit of the tree, an
experiment arm, a release. Keys and values are strings — at most 32 tags, keys of 1 to 64
characters, values up to 256 — anything else is an `InputError`. The record and every exported row
carry them, and so does every tool call the run makes, in `call` ([Tools](#tools)); the model never
sees them. Delegated and routed runs inherit the tags of the run that started them, so a version or
an arm covers the whole tree of runs, and a ticket the tools are bound to binds the delegated run's
tools as well:

```python
run = await engine.run("support", scope="acme", input=message, tags={"commit": "4f2c9e1", "arm": "b"})
run.tags                                             # {"commit": "4f2c9e1", "arm": "b"}
```

Ask for a typed reply and the model is offered an `Output` tool shaped like it. The `expect=` model
is the output's contract, defined in code: its field descriptions and its docstring reach the model
in the tool's schema, and what the output means beyond that is said in `AGENT.md` or a skill. When
the model calls it with input that fits, the run ends and `run.reply` is that input as JSON; input
that does not fit goes back to the model with what is wrong. A model that answers in text instead
ends the run with that text as `run.reply`, which does not parse:

```python
class Verdict(BaseModel):
    refund: bool
    reason: str

run = await engine.run("support", scope="acme", input=message, expect=Verdict)
verdict = Verdict.model_validate_json(run.reply)
```

An accepted `Output` ends the run. A call the model made after it in the same turn is not run; it is
answered with an error saying so, so every call in `run.conversation` has its result. Of a reply
with more than one `Output`, the first that fits is taken: each before it is answered with what is
wrong, each after it is not run, and the reply's other calls before it run as they would without it.

Pass `context=` to add data about this run that does not belong in the tree — the contract under
review, today's date, never instructions — the record keeps it as `run.context` — and pass
`on_event=` to see the run as it happens: where it starts, its reasoning and its text, the messages
it takes from the inbox, its tool calls and their results:

```python
async def show(event):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)

run = await engine.run("support", scope="acme", input=message,
                       context="Acme is on the enterprise plan. Refunds up to $500 need no approval.",
                       on_event=show)
```

A `ToolCallStarted` carries the call's `args`, and a `ToolResultReady` its `step`: the same `Step`
the run records, sent once however the call ended — a result, an error, a refusal, a delegation, a
stop — so a step can be shown, in a conversation or anywhere, while the run goes on. A call
`stop.now()` cuts off sends its `ToolResultReady` whether or not it had started, an `Output`
included; any other call the engine answers without running it sends neither: an `Output` it accepts
or refuses, a call after an accepted `Output`, a call in a reply that was cut short. Its step is in
`run.steps`. Each handler, `approve`, tool and `expect=` model is handed its own copy of a call's
`args` — every list and object copied, any other value as it is — and a handler's `step`, like the
messages a `MessagesTaken` or the adapter is handed, holds copies of the parts, so one that changes
what it is handed changes neither the record nor what the model is sent next.

The same holds wherever the engine hands your code something. A generation model is handed, for each
call, copies of the messages — their parts, their tool calls' `args`, their `reasoning` — and of the
tool specs; a decision model, copies of the state and of the questions; a `Backend`, its own copy of
a command's variables; a storage, rows made for the call; each attempt of a replay, a recorded run
and labels of its own. And a record holds objects of its own: it copies what it is given — the
input, the history, what the inbox held, the questions asked at run time, the tools' specs and
results, each reply's tool calls and reasoning, each answer — and the preload and the questions it
takes from the agent's folder, and a routed run's input is its own copy of its router's state. So no code of yours, and no tenant, reaches another run through what it was
handed or what it returned: changing a record changes nothing the tree shares with every tenant,
nothing of another record and nothing your code holds.

A foreground `Shell` command of a run with a `Workspace` also sends a `CommandOutput` for what
it writes, while it runs, and the calls it makes to the run's tools send their own
`ToolCallStarted` and `ToolResultReady`, with `by` the `Shell` call's id. See the
[Workspace](#workspace) section.

Every model call is a turn, and a `TextDelta` carries the `turn` that said it, from 0. Text of a
turn that ends in tool calls — "Let me look that up." — is said beside them; a run that ends in
text replies with its last turn's text, so `run.reply` is that text and nothing said before it. A
delegated or routed run sends its own `run_id` and counts its own turns.

Every generation run sends a `RunStarted` first, before anything else. Its `call` is the tool call
the run works for — the `Delegate` call that started a delegated run, the call an `engine.tool`
consult was made in, the call that started the decision run a run was routed by — or `None` for a
run started outside any tool call. A frontend places every later event of that `run_id` under that
call, so a sub-agent's text, reasoning and tool calls show nested under the call that asked for
them, at any depth. The event names the call even when it was made in another scope, so a consult
in a shared scope still nests where it was asked; the run's record keeps a `call` only from its own
scope, so no stored row holds another tenant's ids. A
decision run sends no events: the `ToolCallStarted` and `ToolResultReady` of the call that asked it
show it. A `ReasoningDelta` is the model's reasoning as the adapter streams it, with its turn, to show
apart from the reply: it never becomes part of the reply or the conversation, which keeps the
reasoning as the adapter returned it, and reasoning streamed before `stop.now()` is not kept. A model
that does not stream its reasoning sends none.

A person can stop or steer a run while it works. Pass `stop=Stop()` and call `stop.now()` to cancel
the model call in flight and every tool still running, or `stop.after_step()` to let the current
tool calls finish and end before the next model call. Pass `inbox=asyncio.Queue()` and put a string
or a list of parts on it whenever the person says something: at each step boundary — after the
tool results, before the next model call — the run takes everything waiting and appends each as a
`role="user"` message, in order:

```python
stop, inbox = Stop(), asyncio.Queue()
work = asyncio.create_task(engine.run("support", scope="acme", input=message, stop=stop, inbox=inbox))
inbox.put_nowait("Also check the March invoice.")
stop.after_step()
run = await work
run.stopped, run.reply, run.steps
```

When the run takes messages it sends a `MessagesTaken` with them and the turn about to read them.
The inbox is first in, first out and the run takes everything waiting, so a frontend can show a
message as sent but not yet read, and mark the first `len(event.messages)` as read when the event
arrives. Messages still waiting when the run ends were never read and send nothing.

A stopped run returns, never raises: `run.stopped` is `"now"` or `"after_step"` (`None` for a run
that ended on its own), `run.steps` are the steps it took, `run.reply` is the last text the model
produced — text streamed before `now()` included — or `""`, and it is recorded like any run. A tool
call cut off by `now()` is answered with the error `stopped by the user` and its `Step` has
`stopped=True`, so every tool call in `run.conversation` has its result and the next run continues
from it. A call the stop finds not yet running never runs: a stop from a handler of a call's
`ToolCallStarted` comes before its tool runs, and one from a handler of a reply's `TextDelta` before
any of the reply's calls. A handler still running when a stop cancels its call, or a streamed
`TextDelta` or `ReasoningDelta` handler, is cancelled with that work, so a kill switch does its own
work, such as an audit, before it calls `stop.now()`. A call that already has its result keeps it,
and the stop waits for an `on_event` still handling its `ToolResultReady` to return: a handler can
stop the run on what a tool returned; a slow one delays the stop, and cancelling the run cancels it.
The stop waits the same way for a handler of a `RunStarted` or a `MessagesTaken`, which the run sends
between its steps, and one such handler sets ends the run before its next model call. Cancelling the run's
task cancels its model call and every tool call still running, delegations and their runs included,
and re-raises once all of them have ended, as an `asyncio.TaskGroup` does — a call whose tool
returns after the cancel ends with it, before its `ToolResultReady` — and a stop waits the same way
for the work it cancels, so a tool that ignores cancellation holds either one. The first stop is
graceful: a workspace command is sent SIGTERM and given a second to end, and one the workspace is
already stopping on its own — a `Shell` past its timeout, the kill after a `Shell` ended,
`Job(stop=true)` or `close()` — counts as stopped once, so the first stop lets its second run out. A
second stop — `stop.now()` again, or cancelling a run that is already stopping — ends what is still
running at once, with no grace period: each further `now()` cancels once more every tool call still
running but a call whose generation run — a delegated one, or a consult through `engine.tool` —
takes the same `now()`, and a call whose `ToolResultReady` handler the stop waits for; each further
cancel cancels everything still running once more; and a workspace kills its command outright. A
stop set before the run starts ends it before the first model call; one set after it ended does
nothing. A stop reaches delegated and routed runs, and the runs an `engine.tool` starts while the
run calls it, which end the same way at their own next boundary. With `after_step()` pending the
inbox is not read. Whatever is still in the inbox when the run ends stays there — the engine never
starts a run; you do, with those messages as its input. A decision run never reads the inbox: its
questions are one call, and a stop or an inbox rides to the run it routes to; a stop that finds a
delegated or consulted decision run still asking cancels its call, as it cancels a model call in
flight, and that run is not recorded. "Pause, please" is a message in the inbox: the model reads it
at the next boundary and wraps up in its own words.

A run that ended continues from its record, on any process, after any restart. With a refinery
attached every run that returns is recorded with its conversation: keep the run id beside what you
showed — a message's metadata, say — read the run back and pass its conversation on:

```python
last = await refinery.run("acme", run.run_id)
next = await engine.run("support", scope="acme", input="And the second charge?", history=last.conversation)
```

`refinery.run(scope, run_id)` returns the run as `engine.run` returned it, its delegated and routed
runs back in place; a child's id returns that child and the runs under it. A run id the scope does
not have is a `NotFoundError`. A deploy is a stop: on SIGTERM, ask every run in flight to stop after
its step. Each ends at its next boundary and is recorded with `stopped="after_step"`; the next
message, on whichever task takes it, reads the run back and continues:

```python
import signal

running = set()

async def answer(message):
    stop = Stop()
    running.add(stop)
    try:
        return await engine.run("support", scope="acme", input=message, stop=stop)
    finally:
        running.discard(stop)

def shutdown():
    for stop in running:
        stop.after_step()

asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, shutdown)
```

Rewinding is slicing. Cut `run.conversation` just before a `role="user"` message and pass it as the
next run's `history`; files the agent changed in a `Workspace` are yours to restore — a snapshot per
turn, say. The abandoned run stays recorded, and a review of it turns the rewind into preference
data:

```python
turns = [i for i, m in enumerate(run.conversation) if m.role == "user"]
retry = await engine.run("support", scope="acme", input="Refund only the second charge.",
                         history=run.conversation[: turns[-1]])
await refinery.review("acme", run.run_id, Review(stance="objection", correction=retry.reply))
```

### Long conversations

A model keeps nothing between calls: each one is sent the history a run is given, and a
conversation passed back whole grows with every turn — months of it fit no model's context, and
every token of it is paid for again on every turn. The engine does not keep a conversation or
choose what of it to send; you keep it, as you keep any conversation, and tell each run where to
start:

```python
conversation, start = [], 0


def window(conversation, start, turns=20, budget=150_000):
    starts = [i for i, m in enumerate(conversation) if m.role == "user" and i >= start]
    used = next((m.usage for m in reversed(conversation) if m.usage), None)
    if used and used.input + used.cache_read + used.cache_creation > budget and len(starts) > 1:
        return starts[len(starts) // 2]
    return starts[-turns] if len(starts) >= 2 * turns else start


for message in ["My invoice was charged twice.", "And the second charge?"]:
    start = window(conversation, start)
    run = await engine.run("support", scope="acme", input=message, history=conversation, start=start)
    conversation += run.added
```

`start=` is the position in `history` the run begins at; the run sends the messages from there,
and `run.added` — the input and every message after it — is what to keep; a router that routes
nowhere adds nothing, `[]`, so `conversation += run.added` keeps nothing of that turn. `start=0`,
the default, sends the whole history. `run.start` records where the run began, and
`run.conversation` is what the model saw: the history from `start`, then `run.added`. A routed run
is given the history and `start` its router was; a delegated run starts with no history, as always.

`window` is yours, one policy among many. It keeps 20 to 39 turns — a turn is a message the person
sent and everything that answered it — and moves its start 20 turns at a time, so between moves
the history only grows at its end. It starts at a message the person sent, so no tool call is cut
from its result; a history whose sent part holds a result without its call, or a call without its
result, is an `InputError` before any model is called. Each assistant message keeps the `usage`
its call reported, so `input + cache_read + cache_creation` is how many tokens the conversation
had reached there, and the window moves early, keeping the newer half, once the last call passed
`budget`. A window per sitting, per day, by your own count, from a summary you wrote: each is a
function of yours.

A conversation in a database is passed as itself. Anything with `read(start, stop)` that returns
the messages at those positions — every one from `start` when `stop` is `None` — is a
`HistoryLike`, and the run reads it only from `start` on:

```python
class Conversation:
    def __init__(self, pool, conversation_id):
        self.pool, self.id = pool, conversation_id

    async def read(self, start, stop=None):
        rows = await self.pool.fetch(
            "select body from messages where conversation = $1 and position >= $2"
            " and ($3::int is null or position < $3) order by position",
            self.id, start, stop)
        return [Message.model_validate_json(row["body"]) for row in rows]
```

A position is a message's index in the conversation from 0, and always holds the same message.

When a run starts past the beginning, the model is offered `History`, which reads the messages
before the first one it was given, each under its position — a page at a time going back, each
page in the order it happened — or, with `pattern`, the ones whose text matches:

```json
{"before": 340, "limit": 20, "pattern": "invoice", "case_insensitive": true}
```

It never reads what the model was given or anything after it. A page is 20 messages, at most 100;
a search walks back at most `Engine(history_walk=10_000)` messages and a result holds at most
`Engine(history_chars=100_000)` characters of messages, ending with where to go on when there is
more; its deadline is `tool_timeout`, as every tool's, so a slow database gets the time you give
it. No line in `AGENT.md` asks for it: it comes with `start=`, as `tools=` comes with the call. To
send a window without it, pass the slice as the history and leave `start` at 0.

`History` never hands a file back. An image, audio or document shows as a line naming it, with its
URL when it has one — `[image: image/jpeg, https://files.acme.com/a1.jpg]`,
`[document: application/pdf, "rates.pdf"]` — and the model fetches what it wants to see again:
`Load` for a file in the agent's directory, the folders it reads or the workspace, whose call is in
the page above its result, or a tool of yours. A file is paid for on every call that holds it — an
image by its pixels, a PDF by its pages — so one the model has read need not stay in your
conversation either. Keep the line instead, and a tool that brings the file back:

```python
def forgotten(message):
    if isinstance(message.content, str):
        return message
    named = [p if isinstance(p, Text) or p.url is None
             else Text(text=f"[{p.type}: {p.media_type}, {p.url}]") for p in message.content]
    return message.model_copy(update={"content": named})


class SeeArgs(BaseModel):
    url: str
    media_type: str


async def see(args):
    return [Image(media_type=args["media_type"], url=args["url"])]


look = Tools([Tool("See", "Look again at an image by its URL", body=see, args=SeeArgs)])
conversation = [*history, *map(forgotten, run.added)]
```

Pass `tools=look` to each run. `See` hands the vendor the URL to download, so it must be one the
vendor can reach; a document comes back as a `Document` the same way, and a file in your own store
by its data, as `fetch_invoice` does under Tools. A file passed as data has nowhere to come back
from, so `forgotten` keeps it. The turns after a file you swapped send their reasoning no more —
what came before them changed — and the run goes on without it.

Reasoning goes back only where it was given. Some vendors bind a model's reasoning to the exact
conversation it was given in and refuse it anywhere else. Every assistant message the engine makes
from a reply carries `usage`, what the call that wrote it reported, `model`, the model that wrote
it, `settings`, the settings it reported or its adapter declared, and, when the reply carries
reasoning, `prefix`, a short hash of everything sent before it — the model, as its adapter names it,
the system prompt, the tools, the preloaded files and the messages. A message's `reasoning` is sent
again only when what comes before it is still the same; otherwise the message goes without it, its
text and calls as they were, and `run.messages` shows it as sent. So a history you cut or edited, a
changed `context=` or `tools=`, another agent, another model and a replay never send
reasoning a vendor would refuse. A run stopped after a step and continued after a change ahead of
its history — a new date in `context=`, a moved window — sends its last tool call without its
reasoning; a vendor that wants reasoning back on an open tool loop may refuse that, and handling it
is the adapter's job. `prefix`, `usage`, `model` and `settings` are the engine's record; no model
sees them, and `model` says which model wrote each reply when an adapter changes models mid-run,
where the run's `model` names the last. `settings` are not part of the hash: changing them never
takes reasoning away.

Caching is the adapter's. An adapter that marks a cache point has every call of a window read from
the cache what the last call sent; a new window restarts it, and so does any change ahead of the
history — `context=`, `tools=`, a new tree — so a date in `context=` restarts it once a
day.

A history the model cannot take is refused by its vendor: the adapter raises the vendor's error,
`engine.run` raises it as it is, and the run is recorded as far as it got, with its `error` — the
tool calls it already made included. A model that runs out of room while it answers ends the run
with the text it had, recorded like any other with `run.stop_reason` `"max_tokens"`; a tool call
it had begun is answered `NotRun` and never run, so the conversation still passes back. The
window's `budget` is what keeps either from happening.

To measure on a long conversation, tag its runs with it — `tags={"conversation": "dana-1"}` — and
`refinery.replay("acme", candidate, agent="support", tags={"conversation": "dana-1"})` replays them
oldest first, each on what it was sent, with `start=0` and so without `History`. Pass
`conversation=lambda run: Conversation(pool, run.tags["conversation"])` and each replays with the
window it was given and `History` over what came before it, never past its own input.

### Decision agents

A decision model is a discriminative model: given a state and questions whose answers you define,
it returns the probability of each answer — as a generation model is generative, producing text.
What you use the answers for is yours: classify, score, check, extract among candidates, annotate,
grade, route. A decision agent is a directory of questions, asked together in one call:

```
agents/
  triage/
    AGENT.md
    questions/urgent.json
    questions/severity.json
```

A question is `questions/<id>.json`, its id the file's stem: an object with its `type` — `check`,
`score` or `choice` — its `instructions`, and what its answers mean. A `check` may say with `yes`
and `no` what each answer means; a `score` lists its `levels` in order, two or more; a `choice` maps
its `options`, two or more, each key to what it means:

```json
{
  "type": "check",
  "instructions": "Does the message need a person right now?",
  "yes": "The service is down or data is at risk",
  "no": "A question, a request, or a problem with a workaround"
}
```

```json
{
  "type": "score",
  "instructions": "How severe is the problem?",
  "levels": ["Can wait days", "Today", "Within the hour", "Outage"]
}
```

The instructions, what an option means, a level, `yes` and `no` may each be an object or a list
instead of a string, and an option given `null` means no more than its key:

```json
{
  "type": "choice",
  "instructions": {
    "question": "Which team should handle this message?",
    "focus": "The main request, not every topic the message mentions"
  },
  "options": {
    "billing": {
      "what": "Charges, invoices, refunds",
      "not_for": "Delivery or account access",
      "examples": ["I was charged twice", "My refund never came"]
    },
    "orders": {"what": "Delivery, cancellation, returns", "examples": ["My package never arrived"]},
    "other": null
  }
}
```

Each file is checked when the tree loads, so a mistake is caught before any run: a file that is not
a JSON object, a key a question does not take (`tree: agents/triage/questions/urgent.json has the
unknown key 'option'; a question takes type, instructions, options, levels, yes and no`), a key
of another type, such as `levels` on a choice (`tree: agents/triage/questions/team.json is a choice
question holding 'levels'; a choice takes options, a score levels and a check yes and no, so drop
'levels' or change its type`), instructions missing or empty, a choice with fewer than two options
or a score with fewer than two levels is a `ConfigurationError` naming the file. A question in
markdown, `questions/<id>.md`, is one too, which says to write it as `<id>.json`.

Run it with a state: a string, an object or a list.

```python
engine = Engine(root=Path("agents"), models={"judge": dm})

run = await engine.run("triage", scope="acme", input={"message": text, "plan": "enterprise"})
run.answers["urgent"].probability     # 0.91
run.answers["severity"].score         # 3.0
run.answers["severity"].confidence    # 0.87
```

Every question is answered in the same call, in parallel and in isolation: adding a question
changes neither the others' answers nor, noticeably, the latency. An engine with only a decision
model needs no `Tool`s; the tree's generation agents are the ones it cannot run.

Some answer spaces live in your data: the orders of the customer who wrote, the plans of this
tenant. A script beside the question fills it: `questions/<id>.py`, a function `run(input, call)`
that returns the fields it fills — `{"options": {...}}` for a choice, `{"levels": [...]}` for a
score — merged over `<id>.json`, which leaves them out. The JSON fixes the rest, its `type` always:
a script never changes what kind of question it fills, nor anything its JSON sets, so every run
asks the same question under one id and only what the script fills differs between tenants:

```
agents/
  matcher/
    AGENT.md
    questions/order.json
    questions/order.py
```

```json
{"type": "choice", "instructions": "Which of the customer's orders is the message about?"}
```

```python
from json import load
from os import environ
from urllib.request import urlopen

ORDERS = environ["ORDERS_URL"]


def run(input, call):
    url = f"{ORDERS}/{call['scope']}/customers/{input['customer']}/orders"
    with urlopen(url, timeout=10) as reply:
        return {"options": {order["id"]: order["summary"] for order in load(reply)}}
```

```python
async with Engine(root=Path("agents"), models={"judge": dm}) as engine:
    run = await engine.run("matcher", scope="acme", input={"message": text, "customer": "c-17"})
    run.questions["order"].options      # {"A-1001": "Standing desk, shipped", ...}
    run.answers["order"].choice         # "A-1001"
```

The script runs in the decision agent's worker, as a tool's does ([Tools](#tools)), before the
decision model is called. `input` is the run's state as JSON — the text, the object or list as
given, or the parts as their JSON, media `data` in base64 — and `call` holds the run's `scope`,
`run_id`, `agent` and `tags` and the `question` it fills, so what it reads is the tenant's. The scripts of
a run's questions run at once, each under `tool_timeout`, and the question each fills is checked as
a file is. A script that raises (`WorkerError: engine: order.py of decision agent 'matcher' could
not fill question 'order': KeyError: 'customer'`), runs past its deadline (`TimeoutError: workers/pool:
order timed out after 30s`), returns anything but an object, returns a key its JSON sets
(`ModelError: engine: order.py of decision agent 'matcher' returned 'type', which question 'order'
sets in order.json; a script returns only what its question's JSON leaves out, such as {"options":
{...}}`) or one its type does not take, or fills a question that does not fit (`ModelError: engine:
question 'order' of decision agent 'matcher' does not fit once order.py fills it: the question:
models: a choice needs at least two options`) ends the run before the decision model is called,
recorded with its `error` as a failed decision call is, and with only the questions that need no
script. Otherwise the record keeps each question as it was asked, the filled options included, and
`fingerprint.questions` follows them: two tenants' runs of the matcher above ask the same question
with different options, under two question fingerprints and one agent fingerprint. A
question's script imports what a tool's does — the standard library, installed packages and the
helpers of its folder, `questions/_name.py`, at the top of the script — and is read when the tree
loads, as a tool's is, so `fingerprint.agent` covers it; a `.py` without its `.json`, one with no
`run` at its top level, or a helper named like a standard-library module, `questions/_stat.py`, is a
`ConfigurationError`, as it is for a tool's helper ([Tools](#tools)). A route goes by a question
whose options its JSON fixes, never by one a script fills
([Both kinds together](#both-kinds-together)).

Some answer spaces are known only when the run starts: the children of the node a walk chose last,
the candidates a search returned, the skills an agent has today. Those questions are composed when
the run starts, by a model or by a runtime of yours: a generation agent composes them
in a `Delegate` call ([Nesting](#nesting)), and a runtime of yours composes them in a
call to `engine.tool`. They are ids to `Choice`, `Score` or `Check` — or anything
with their fields — written as JSON, and they are asked with the folder's,
in the same call. A question known in advance lives in `questions/`, and one whose answers your
data holds keeps its script there. The record keeps every question asked, and `run.runtime` names
the ids that came at run time:

```python
asking = engine.tool("triage", scope="acme")
sent = {
    "input": text,
    "questions": {
        "product": {
            "type": "choice",
            "instructions": "Which product is the message about?",
            "options": {
                "desk-120": {"what": "Standing desk, 120 cm"},
                "desk-160": {"what": "Standing desk, 160 cm"},
            },
        }
    },
}
result = await asking.dispatch("triage", sent)
result["answers"]["product"]["choice"], result["answers"]["urgent"]["probability"]
```

A runtime question holds what a question file holds — a choice needs two options, a score two
levels — and one that does not, an id the run already asks, or questions sent to a generation
agent is an `InputError`. A decision agent may have no `questions/` at all and ask only what each
run brings; a run with nothing to ask is an `InputError`.

Every run of a decision agent asks its `questions/`, and beside them the questions the run brings;
another set of questions, asked on other runs, is another decision agent, with its own folder and
its own name, under which its records and its datasets stand apart.

`run.fingerprint.questions` is a short sha256 over the questions the run asked — the folder's, those
its scripts filled and the runtime ones alike — as JSON ordered by question id, each question's
fields as they were asked and its options and levels in their order, since the order can change
which answer a model favours. Two runs that asked the same questions share it; a changed word,
option or order changes it. A generation run's is `None`. With `fingerprint.agent` for the folder,
it tells two versions of a decision apart even when the questions came at run time.

The state is text unless the decision model reads more. An adapter that carries `media_types` —
patterns such as `{"image/*", "application/pdf"}`, as a generation adapter's — takes a list of parts
with that media as the state, parts as they are, and the record keeps them. A list of parts made
only of `Text` is joined with newlines into the state, as it is for an adapter without
`media_types`, which reads text only. A part the adapter does not declare is an `InputError` before
the model is called, naming what it takes (`engine: decision agent 'triage' takes text only; got
image/png`).

Everything beyond the answers — weighing several, confirming with a person below a threshold,
acting — is your code over the returned `Decision`. The library gives you the answers; it does
not make the call.

### Nesting

`agents:` in a generation agent's frontmatter gives it a `Delegate` tool listing those agents by
name, description and kind. The model calls it with an agent and an input; the child runs
under the same scope and its reply — text for a generation agent; for a decision agent its answers,
and the routed run's reply when it routed, as `engine.tool` returns them — comes back as the tool
result. Its input comes from the model and is never read for `/name`, and the child takes no skill.
The child's record is attached to the parent's `children`, and the
step that called it carries the child's `run_id`. A decision agent used this way is how a
generation agent asks a fast, calibrated question mid-run: is this destructive, which skill
applies, is the user done.

The model can also give a decision agent `questions` it composes itself — ids to questions of the
three types, written as JSON, structure included — asked beside the agent's own in the same call. A
question that occurs to the model mid-run is answered with probabilities instead of guessed at, and
a decision agent with no questions of its own is worth delegating to. The model sends:

```json
{"agent": "triage", "input": "The March invoice shows the same $49 charge twice.",
 "questions": {"refund": {"type": "check", "instructions": "Does the customer ask for money back?"},
               "cause": {"type": "choice", "instructions": "What caused the second charge?",
                         "options": {"duplicate": "The same payment taken twice",
                                     "plan_change": "A prorated change of plan", "other": null}}}}
```

```python
run = await engine.run("support", scope="acme", input="I was charged twice in March. Refund one.")
child = run.children[0]
child.runtime                             # ["refund", "cause"]
child.answers["refund"].probability       # 0.91
child.answers["cause"].choice             # "duplicate"
```

A decision child records its runtime questions, as any run does, and the step keeps what the model
sent. `Delegate` offers `questions` only when a target is a decision agent, and its schema describes
the question object once, for all three types. Questions for a generation agent, a question that
does not fit, an id the target already asks, a decision run with nothing to ask, or a question
script that fails reaches the model as a tool error it can read and act on, and the run goes on.

A run records the call it worked for in `call`, the same `CallInfo` its `RunStarted` carries, so a
run a tool of yours started links back to the run that called it.

Depth is capped (`max_depth`, default 5). Within it an agent may name itself in `agents:` and hand a
sub-problem to a fresh run of itself, and two agents may call each other. A route is different: it
forwards the same input, so a route into an agent that already decided it — in the same chain of
routes, since the last `Delegate` — is refused with `DelegationLoopError`; a `Delegate` starts a new
input, so the agents above it do not count. Through `Delegate` an error reaches the model as a tool
error and the run continues; through a route it raises.

### Both kinds together

The two kinds call each other. A generation agent asks a decision agent through `Delegate`, as in
[Nesting](#nesting), and a decision agent can hand its run to another agent by one of its answers:
a route. One choice question in `questions/`, one table from its options to agents, an optional
confidence floor:

```
agents/
  frontdesk/    kind: decision   model: judge   route: {by: intent, to: {billing: billing, technical: support}}
  support/      kind: generation model: strong  agents: [triage]
  billing/      kind: generation model: strong
  triage/       kind: decision   model: judge   questions: urgent (check), severity (score)
```

```markdown
---
name: frontdesk
kind: decision
model: judge
description: Routes an incoming message
route:
  by: intent
  to: {billing: billing, technical: support}
  min_confidence: 0.6
---
```

```json
{
  "type": "choice",
  "instructions": "Which department should handle this message?",
  "options": {
    "billing": "Payments, refunds, invoices",
    "technical": "The product misbehaves",
    "other": "Anything else"
  }
}
```

When the chosen option maps to an agent and confidence clears the floor, that agent runs with
everything the run was given — the same input, parts included, `history`, `context`, `expect`,
`tools`, `workspace`, `on_event`, `stop`, `inbox` and `tags` — and the routed run is attached:

```python
engine = Engine(root, models={"strong": llm, "judge": dm})

run = await engine.run("frontdesk", scope="acme", input=message, history=history)
run.answers["intent"].choice            # "technical"
run.routed_to.agent                     # "support" — the generation agent that answered
run.routed_to.children[0].agent         # "triage" — the decision agent support consulted mid-run
```

A route is by a question of the router's `questions/` whose options its `.json` fixes, since the
route maps them to agents: neither a runtime question nor one a script fills can drive it, and `by:`
naming a scripted question is a `ConfigurationError` (`tree: route of 'triage' is by 'order', whose
options come from order.py; route by a question whose options are fixed`). A router's runtime
questions are its own, asked beside `intent` and not passed on. A route entry is an agent's name, as
in `to:` above, and a decision agent a route reaches must have `questions/` of its own; anything
else is a `ConfigurationError` when the tree loads, since that tree cannot work. A decision agent
reached through `Delegate` needs no questions of its own, since the model can bring them.

One message, four model calls, one record. Given only `strong` the same tree runs its generation
agents and refuses the decision ones; given only `judge` the reverse, and a decision agent whose
route reaches an agent whose model was not given is refused before its questions are asked. Nothing
about an agent changes when a model is absent — only which agents you can run.

### Orchestrating

The library runs agents and records what they did. The order they run in is yours or the model's: your
code decides in a workflow, and the model decides when it delegates, routes or calls a tool of yours.
Start with a workflow in code, where you know the steps, and add autonomy only where a fixed step
breaks. There is no graph, planner or queue here to configure; what follows is what you write on
top of the public surface, and every block runs in the test suite.

A workflow is code. Tag every run with the workflow's id, so `run_tree(tags=)` and
`delete_runs(tags=)` reach all of it:

```python
workflow = {"workflow": "refund-17"}

billed = await engine.run("billing", scope="acme", input=message, tags=workflow)
answered = await engine.run("support", scope="acme", input=billed.reply, tags=workflow)

parts = await asyncio.gather(
    *(engine.run("support", scope="acme", input=part, tags=workflow) for part in (message, text))
)

routed = await engine.run("frontdesk", scope="acme", input=message, tags=workflow)
routed.routed_to.agent                  # the department the router chose
```

One run after another, runs side by side, and a route: the three patterns a workflow is made of.

A budget across a tree is an `on_event` that counts `RunStarted`, which every generation run in the
tree sends, and presses a `Stop` when the count runs out. Decision runs send no events, so they are
not counted:

```python
class Budget:
    def __init__(self, runs):
        self.left, self.stop = runs, Stop()

    async def __call__(self, event):
        if isinstance(event, RunStarted):
            self.left -= 1
            if self.left < 0:
                self.stop.after_step()

budget = Budget(50)
run = await engine.run("support", scope="acme", input=message, on_event=budget, stop=budget.stop)
```

Two agents that write `guide.md` into one workspace overwrite each other. Give each run a copy of
its own, have it claim work before it does it, and merge when a child comes back. A `Backend` is
yours, so the copy is a directory per run, named by `current_call().run_id`: inside a tool or a
backend, `current_call()` is the call the running run is making. `RunStarted.call` names the run that
started a child, and the parent's `ToolResultReady` for its `Delegate` step carries
`step.child_run_id`, which is where you merge. How a directory is copied and merged (worktrees, a
copy, a container per run) is yours too:

```python
from pathlib import Path

from rfnry_engine import Local, RunStarted, Tool, ToolResultReady, Tools, Workspace, current_call

class PerRun:
    def __init__(self, base):
        self.base, self.parent, self.merged = Path(base), {}, []

    def here(self):
        call = current_call()
        place = self.base / (call.run_id if call else "main")
        place.mkdir(parents=True, exist_ok=True)
        return Local(place)

    async def spawn(self, command, env): return await self.here().spawn(command, env)
    async def read(self, path, limit): return await self.here().read(path, limit)
    async def write(self, path, data): await self.here().write(path, data)

    async def on_event(self, event):
        if isinstance(event, RunStarted) and event.call:
            self.parent[event.run_id] = event.call.run_id
        if isinstance(event, ToolResultReady) and event.step.child_run_id:
            self.merged.append(event.step.child_run_id)

claims = {}

async def claim(args):
    me = current_call().run_id
    return {"claimed": claims.setdefault(args["item"], me) == me, "by": claims[args["item"]]}

backend = PerRun("work")
tools = Tools([Tool("claim", "Take an item of work for this run", body=claim, input_schema={
    "type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]})])
run = await engine.run("lead", scope="acme", input=message, tools=tools, on_event=backend.on_event,
                       workspace=Workspace(backend, scope="acme"))
```

Work in the background is two tools of yours: one starts `engine.run` in a task you keep, one reads
the task. You cancel these tasks and bound how many there are. The run they start records the call
that started it, so its record links back to the run that asked for it:

```python
jobs = {}

async def start(args):
    jobs[str(len(jobs))] = asyncio.create_task(
        engine.run("support", scope="acme", input=args["input"])
    )
    return {"job": str(len(jobs) - 1)}

async def check(args):
    job = jobs[args["job"]]
    return {"running": not job.done(), "reply": None if not job.done() else job.result().reply}

tools = Tools([
    Tool("start", "Start support on an input", body=start, input_schema={
        "type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}),
    Tool("check", "Read a job started earlier", body=check, input_schema={
        "type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
])
outer = await engine.run("boss", scope="acme", input=message, tools=tools)
```

Whether a team is worth it is measured, not assumed. Run the team and a single agent on the same
input and compare what each spent over the whole tree, summed over `children` and `routed_to`. A
run a tool of yours started as a consult is not a child: it links back to its caller through its
`call`, and `spent` does not count it. Over the holdout, with the grades beside the cost, the same
comparison is a replay of each, as [`evaluation.md`](evaluation.md#replaying-the-holdout) shows:

```python
def spent(run):
    below = run.children if isinstance(run, Generation) else [run.routed_to] if run.routed_to else []
    usage, calls = run.usage, run.calls if isinstance(run, Generation) else 1
    for child in below:
        more, made = spent(child)
        usage, calls = usage + more, calls + made
    return usage, calls

team, team_calls = spent(await engine.run("support", scope="acme", input=message))
alone, alone_calls = spent(await engine.run("billing", scope="acme", input=message))
more = team_calls - alone_calls, team.output - alone.output    # calls and tokens the team adds
```

### Workspace

> `Workspace` gives the model a real shell, and **a shell is not confined by its directory**.
> `Local` runs each command as a process of your service: without a sandbox in front of its shell, a
> command can do anything the service's user can — read every other tenant's workspace, read the
> service's configuration and any secret on disk, reach the database and the network, delete things.
> A document an agent reads can tell it to try. Put a sandbox in front of the shell, as below, or
> pass a backend that runs commands somewhere else, and keep the agent directory outside the
> workspace.

A workspace is where an agent writes code, runs it, reads what happened and tries again — the loop
a coding assistant runs in a developer's terminal. An agent opts in with one line in its
`AGENT.md`, and a run supplies the workspace: the library's tools over a backend of yours, which
runs the commands and holds the files.

```
agents/coder/AGENT.md
---
name: coder
kind: generation
model: strong
description: Writes and runs code to finish the work.
workspace: true
---
```

```python
from rfnry_engine import Local, Workspace

local = Local("/srv/tenants/acme/work", shell=[
    "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
    "--tmpfs", "/tmp", "--bind", "/srv/tenants/acme/work", "/srv/tenants/acme/work",
    "--chdir", "/srv/tenants/acme/work", "--unshare-all", "--die-with-parent", "--new-session",
    "bash", "-c",
])
workspace = Workspace(local, scope="acme", timeout=600)

run = await engine.run("coder", scope="acme", input="make the failing test pass", workspace=workspace)
```

`Local` runs each command on this machine, in the directory, given to the words of `shell` — here
bubblewrap, which builds each command a world of its own from the host: the system's programs
read-only, this tenant's directory writable, an empty `/tmp`, and nothing else — no other tenant,
no service configuration, no network — and kills every process when the command ends. The model
sees none of this: it has the same tools, and its command runs inside.

A sandbox per command, as here, gives each command its own network, so a server a job starts
answers only commands inside the same sandbox; to start a server in one message and reach it in
the next, keep one sandbox per workspace, such as a container kept for the workspace's life behind
a backend of yours, below.

The agent then has five more tools. `Shell(command, background=false)` runs a command and returns
`exit_code`, `stdout`, `stderr` and `timed_out`; when the command ends, anything it left running is
stopped. With `background=true` the command starts as a job — a server, a watcher, a long build —
and the call returns its id at once, `{"job": "j1"}`. `Job(id, wait=0, stop=false)` returns what the
job printed since the last read, stdout and stderr in the order they arrived, whether it is
`running`, and its `exit_code` once it has ended; `wait` is seconds to wait for the job to end
first, up to the workspace's `timeout`, returning as soon as it does — a model waits for a build
instead of asking again and again — and `stop=true` stops a job still running after the wait.
`Jobs()` lists the jobs: id, command, when it started, running or exit code. `Write(path, content)`
writes a file exactly, and `Edit(path, old, new, all=false)` replaces one exact occurrence, or with
`all=true` every one — helpers that send only what changes and leave a clean step in the record; the
shell can do both too. `Read` and `Load` take `where="workspace"`: `Read` for a text file of the
workspace, with `offset` and `limit` selecting lines, `Load` to hand the model a file as it is, so
an agent looks at the chart it just plotted. Both go through the backend. `LS`, `Grep` and `Glob`
stay on the folder; a path the folder does not hold, given to `LS`, `Read` or `Load` of a run with a
workspace, is answered with the way to the workspace: `tools/builtins: not a directory of your folder:
task_folder; Shell lists your workspace`. Each command starts fresh
in the directory, so a `cd` or a variable does not carry over, and the tool's description tells the
model so, along with its limits.

Jobs belong to the workspace, not to the run: they keep running after the run that started them,
after a run that failed and after `stop.now()`, until the model stops them or you close the
workspace — so an agent starts a server in one message and tests against it in the next. Every run
given the workspace, delegated and routed runs included, sees the same jobs, numbered `j1`, `j2`, …
for the workspace's life. A job that ended stays listed until its last output is read. Each job is
read in the background until it ends or the workspace is closed, so it never stalls on a full pipe;
what it printed waits for `Job`, kept to its first and last `output_bytes // 2` bytes. The list of
jobs is not in the system prompt, which would change with every job and cost the cached prefix of
every conversation on the workspace; the model asks `Jobs()`.

A frontend shows the same jobs with `workspace.jobs()`, a list of `JobInfo(job, command, started,
running, exit_code)`, and reads one with `workspace.output("j1")`, everything the job printed so far
in the same cap. Neither changes what the model's `Job` reads next, and a job that ended leaves the
list once the model reads its last output. An unknown id is an `InputError`.

Close the workspace when you are done with it — `async with Workspace(…) as workspace:` or `await
workspace.close()` — and every job is stopped; the workspace stays usable. A workspace never closed
leaves its jobs to the backend: on `Local` they may outlive your process, unless the sandbox ties
them to it, as bubblewrap's `--die-with-parent` does. A workspace and its jobs live in one event
loop, and a job whose loop ends is stopped: a service keeps a conversation's workspace in one of its
processes and on one loop.

`Workspace(backend, scope=, timeout=, output_bytes=, bridge=, tree=)`:

| argument | default | what it does |
| --- | --- | --- |
| `backend` | | where commands run and files live: `Local`, or one of yours |
| `scope` | | the tenant the workspace belongs to |
| `timeout` | `120` seconds | the limit on each `Shell`, `Job`, `Write` and `Edit` call; a `Shell` command still running is stopped — on `Local`, sent SIGTERM, then killed a second later with everything it started — and `Shell` returns what it printed with `timed_out: true` |
| `output_bytes` | `30_000` | the cap on `stdout` and on `stderr`, each keeping its first and last half |
| `bridge` | `True` | lets a command call the run's tools, below; `False` turns it off |
| `tree` | `None` | where the agent tree is mounted in the sandbox, as commands see it, below |

`Local(root, shell=, env=)`:

| argument | default | what it does |
| --- | --- | --- |
| `root` | | the directory commands start in and files live in |
| `shell` | `["bash", "-c"]` | the words each command is given to, last: a sandbox goes in front of the shell, and the first word must be on `PATH` |
| `env` | `PATH`, `HOME` (the directory), `LANG` | the only environment commands see, so the service's own keys never reach them |

The workspace's `timeout` replaces `Engine(tool_timeout=)` for `Shell`, `Job`, `Write` and `Edit`,
so a ten-minute build and a thirty-second API call live in the same engine. A `shell` whose first
word is not on `PATH` is a `ConfigurationError` that says so; pass `shell=["sh", "-c"]` on a system
that has only a POSIX shell, such as Alpine.

On `Local`, `Write`, `Edit`, `Read` and `Load` stay inside the directory: they refuse a path that
leaves it, symlinks included. What the system refuses reaches the model with its reason and the
model's own path, never the path on your machine: `tools/workspace: cannot write notes.txt/x: Not a
directory`. `Shell` reaches what the process running it can reach, so the real boundary is the
sandbox in front of the shell, or the machine. `Local` reads and writes files in a thread, which a
cancel cannot stop, so a `Write` or `Edit` cancelled while it writes returns once the file is
written, and one past the workspace's `timeout`, like a workspace `Read` or `Load` past
`tool_timeout`, returns its `TimeoutError` only once its thread has ended.

The workspace root and the folders above it must be out of the model's reach: `Local` checks every
call against the folder it was made with, but a `Local` made later on a path the model could rename
follows the rename. The bubblewrap recipe binds the root itself, so a sandboxed model never sees
its parent.

`Local` refuses a root that is no longer the folder it was made with, by its directory inode number.
On a filesystem whose directory inode numbers are not stable — overlayfs without `xino`, FUSE
without `use_ino` — that check refuses a valid root, so such a filesystem is not supported as a
workspace root.

Binding the whole system read-only (`--ro-bind / /`) is not enough: a command could still read
every other tenant's directory and every secret on disk. Mount only what commands need — the
recipe above suits a system whose programs live under `/usr`; add a read-only `/etc` path when a
tool needs one — and give each tenant its own workspace, each with its own sandbox. bubblewrap
limits nothing: one command can take all the memory, CPU or processes of the machine from every
other tenant. Where the service's user has a systemd user session,
`systemd-run --user --scope -p MemoryMax=1G -p CPUQuota=100% -p TasksMax=256 --quiet` before
`bwrap` runs each command in a scope with those limits; `env=` then needs `XDG_RUNTIME_DIR` too,
which `systemd-run` uses to find the session.

Docker can be the sandbox too. A container per command is a `shell=` of `docker run --rm` with the
directory mounted at its own path and `timeout -s KILL <seconds>` before the command inside: the
`docker` client passes SIGTERM on to the container, but SIGKILL stops only the client, so the limit
inside is what ends a container whose command ignores SIGTERM. `env=` is then the client's
environment, not the command's — the command's own variables go in with `-e`: a workspace with
`tree=` needs `-e RFNRY_AGENT -e RFNRY_TREE`, and one with the tool bridge, below,
`-e RFNRY_TOOLS -e RFNRY_TOOLS_KEY` — and the client reads
its configuration from `$HOME/.docker`, which `Local`'s default `HOME` puts in the workspace, where
a command could write a `.docker/config.json` the next command's client obeys; point
`DOCKER_CONFIG` outside the workspace. One container per workspace, entered with `docker exec`,
needs a backend of yours: signals do not pass through `docker exec`, so its `kill()` must reach the
command's processes inside the container itself — a tag in their environment that everything they
start inherits, say, whose processes it signals.

The folder in the shell. `Workspace(local, scope="acme", tree="/srv/agents")` says where the agent
tree is mounted in the sandbox, as commands see it. The commands of a run then get `RFNRY_TREE`, the
tree as `posixpath.normpath` writes it, so `/srv/agents/` is `/srv/agents`, and `RFNRY_AGENT`, the
running agent's folder in it. The `Shell` description tells the model that the folder is at
`$RFNRY_AGENT`, read-only, with its files, skills and scripts to read and run, and that each folder
the agent reads, or that its `files.include` names, is at `$RFNRY_TREE/<name>`. A skill's script is run as
`python "$RFNRY_AGENT/skills/plot/run.py"`. The sandbox mounts the tree read-only at the same path:
`--ro-bind /srv/agents /srv/agents` for bubblewrap, `-v /srv/agents:/srv/agents:ro` for Docker, and
for an unsandboxed `Local` the tree's own path. Commands started with `background=true` get the
variables too; a `Shell` that code dispatches outside a run gets none. Cautions:

- the shell sees every file under the mount, `rfnry.yaml`, hidden ones, those `gitignore: true`
  leaves out and whatever `files` leaves out included, which the file tools refuse;
- the fingerprint covers what the folder holds, not what a command reads;
- a mount of the whole tree shows the other agents' folders; mount less to hide them;
- `refinery.replay(candidate, workspace=)` needs a workspace whose sandbox mounts the candidate's
  tree.

Showing a command. With `on_event`, a foreground `Shell` command sends a `CommandOutput` for what it
writes, as it arrives (chunks that queued while the handler was busy arrive joined, one event per
stream, and one more at each gap, since an event never joins across one), with its `call_id`, the
`stream` (`"stdout"` or `"stderr"`) and the `text`, decoded as UTF-8 per stream so a character split
across two writes arrives whole. A frontend can show a long build as it runs instead of after it.
The model's view is unchanged: it reads the same capped `stdout` and `stderr` in the `Shell` result,
and a handler never slows the command. The events wait in a queue of 1 MiB per command; while a
handler is busy past that, chunks are dropped from the events, as is a single read of more than 1
MiB even while the handler is idle, and once a stream overflows its output is dropped until the
handler catches up and takes what is queued; the next event of that stream starts with `[... <n>
bytes not shown ...]`, counting all of it. Every event of a command comes before the
`ToolResultReady` of its call, in order per stream — `stdout` and `stderr` are two pipes, with no
order between them — and a handler that raises fails the run once the command has ended. A command
started with `background=true` sends none.

Calling tools from code. A command can call the run's tools itself, so a script that looks up fifty
orders, or checks every row of a table against a rule, makes its calls in a loop, and the model reads
only what the script printed instead of making one call a turn. For each foreground `Shell` command
of a run, the engine listens on a Unix socket in the workspace and gives the command its path and a
key of its own:

```
 model                    engine (your process)                           sandbox (this tenant's shell)
   │ Shell("python solve.py")    │                                                 │
   │────────────────────────────▶│ 1. listen on <workspace root>/.rfnry-<8 hex>.sock
   │                             │ 2. spawn the command with RFNRY_TOOLS=<that path> ──▶ python solve.py
   │                             │◀── {"key":…,"tool":"search","args":{"query":"heat equation"}}   (connect, one line)
   │                             │ 3. the run's call path: events, approve, deadline, dispatch, record
   │                             │─── {"result":[ … ]} ─────────────────────────────────────▶│ (one line, close)
   │◀── Shell result ────────────│ 4. command ended: socket closed, removed if its path still holds it
```

A client is a few lines in any language; in Python:

```python
import json, os, socket


def call(tool, **args):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(os.environ["RFNRY_TOOLS"])
        request = {"key": os.environ["RFNRY_TOOLS_KEY"], "tool": tool, "args": args}
        s.sendall(json.dumps(request).encode() + b"\n")
        return json.loads(s.makefile(encoding="utf-8").readline())


order = call("lookup_order", order_id="A-1001")
print(json.loads(order["result"])["status"])
```

A connection makes one call: one line of JSON in, `{"key": …, "tool": …, "args": {…}}`, and one
line back, `{"result": …}` or `{"error": …}`, then it closes. `result` is what the model would read
for the call: its text as the step records it — a tool that returns an object or a number gives its
JSON text, so a script runs `json.loads` on a result it expects to be JSON — or, for a tool that
returns parts, their JSON list, media `data` in base64. `error` is the text the model would read:
`NotApproved: lookup_order was not approved`, `TimeoutError: engine: lookup_order timed out after
30s`, a tool's own error. A line that is not one JSON object with a string `tool` and an object
`args`, a line over 1 MiB, and a request without the command's key are answered with an `error` that
starts with `tools/bridge:`, and run nothing. A command makes at most 10,000 calls and sends at most
2 MiB of request lines in all: every request that reaches the run's tools is a call, a name the
socket does not carry included, and every line read whole counts toward the 2 MiB as it is read,
before it is parsed, one the bridge refuses included. Past either, every further request is answered
`{"error": "tools/bridge: this command made its 10000 calls; no more are answered"}` or
`{"error": "tools/bridge: this command sent its 2097152 bytes of calls; no more are answered"}`, so
a script with more to do splits it over several commands, as the `Shell` description tells the
model. The budgets are each command's own: commands a run has going at once each get theirs, so
what a run's commands send in all grows with how many it runs at once. A call whose `args` nest
deeper than 128 levels of lists and objects, `args` itself the first, is answered
`{"error": "tools/bridge: a call's args nest deeper than 128 levels"}`. The depth is read from the
line before it is parsed, its strings aside and the request itself the first level, so any field
nested past 129 levels gets this error, not only `args`. A request is checked in this order and
answered with the error of the first check it fails: a line over 1 MiB, which is not read whole
and counts toward neither budget; then, the line counted toward the 2 MiB, the 10,000 calls and
then the 2 MiB; then the depth; then a line that is not one JSON object; then the command's key;
then a string `tool` and an object `args`. So a line over 1 MiB gets the size error whatever else
is wrong with it, a line past a budget gets the budget's error even when it is nested too deep or
malformed, and a line nested past 129 levels gets the depth error even when its key is not the
command's or it is no JSON at all. None of these runs anything or leaves a step or an event, and a
connection closed at the cap of 64 below, refused before its line is read, counts toward neither
budget.

The socket carries every tool the run offers the model — the builtins, `History`, the folder tools,
the run's `tools=` — except the workspace's own (`Shell`, `Job`, `Jobs`, `Write`, `Edit`) and any
tool named `Output` or `Delegate`, the engine's or one of the run's `tools=`, which the model still
calls. A name the socket does not carry gets the model's error for an unknown tool,
`InputError: tools: unknown tool 'Shell'`, and leaves no step and no event. Connections are answered
as they come, so calls a script makes at once — from threads, or with `asyncio` — run at once, as a
reply's calls do, up to 64 at once per command: a connection that arrives while 64 are in flight is
closed at once, unread, after an `{"error": "tools/bridge: at most 64 calls at once; wait for one to
finish"}` line. The script may read that line or see a broken pipe or a connection reset instead, and
should retry on any of them. A `Shell` command of a delegated or routed run calls that run's tools,
under its own `run_id`.

Each call runs as the model's calls do — `ToolCallStarted`, `approve`, the deadline (`tool_timeout`,
or a folder tool's own), the stop, `ToolResultReady` — and is recorded as a `Step` with `by`, the id
of the `Shell` call whose command made it; its own id is `<Shell call id>.<n>`, n counting from 1 for
each command. It finished first, so its step sits in `run.steps` before the `Shell` step that made
it, and the conversation holds no message for it: the model reads only the `Shell` result.
`ToolCallStarted`, `ToolResultReady` and `CallInfo` carry the same `by`, so `approve` and a tool tell
a call a script made from one the model made with `current_call().by`.

A call ends with its command. When the command ends, for any reason — it exits, a stop, the
workspace's `timeout` — a call still in flight is cut as a stopped call is: its step is recorded with
`stopped` and `stopped by the user` as its result, its `ToolResultReady` is sent, and a script still
connected reads `{"error": "stopped by the user"}`. A consult that a script's call started, an
`engine.tool` passed in `tools=`, ends with the command: one still running when the command ends is
cancelled and leaves no record of its own, as no cancelled run does. The exception is a command that
`stop.now()` ended: its process is killed at once, while the consult takes the stop, as one the
model's own call started does, and keeps its record, and the `Shell` call ends after it. A further
`stop.now()` does not cut it; a cancel of the run, or its call's deadline, does. The script's step
is recorded stopped either way. A reply to a process that outlived the command, one that left its
process group outside a sandbox, is written but not waited for, and may be cut off when the socket
closes. An `on_event` that raises on a script's call fails the run once the command has ended.

The socket is the one file the engine creates itself, `<root>/.rfnry-<8 hex>.sock`, inside the
workspace you passed, for the life of one command: made just before the command starts, closed once
it has ended and its output is read, and then removed if its path still holds it — a file the
command put in its place is kept, and a socket in a directory the command made read-only stays.
Under bubblewrap it is in the bound directory at the same path, and a socket bound to a path reaches
through `--unshare-all`; under Docker it is in the directory mounted at its own path, with
`-e RFNRY_TOOLS -e RFNRY_TOOLS_KEY`. The socket can be reached only from a sandbox on the same kernel
that sees the workspace directory at the same path: bubblewrap, or Docker on Linux. Docker Desktop's
file sharing on macOS and Windows does not carry Unix sockets, so use a backend that runs the engine
beside the commands there. A container user other than the service's needs write permission on the
socket file, which takes the service's umask and a new random name for each command, so it cannot be
granted ahead of time: run the container as the service's uid, or as a user of the service's group
with the service under a umask of 002. The `Shell` description tells the model of it in those
variables, never in a path. Whether the bridge is on, and whether `tree=` is set, change that
description and so `fingerprint.tools` — rightly, since the model was offered something different —
but where a deployment puts the workspace does not.

- The bridge gives code nothing more than the model has: the model's tools but those named above,
  through the same `approve`, scope and record. What is new is speed: a script can make many calls
  quickly; `approve` still sees each one.
- Only processes that see the socket's path can connect — with a sandbox per tenant, only this
  tenant's commands — and only a request with the command's key is answered.
- The key is as private as the sandbox keeps a command's environment: a process of the same uid in
  the same PID namespace reads it from `/proc/<pid>/environ`. A sandbox per command, as the
  bubblewrap recipe above runs each one under `--unshare-all`, keeps it to that command. No
  sandbox, one container per workspace entered with `docker exec`, or anything else where commands
  share a PID namespace and a uid does not: there another command, a background job included, can
  read a running command's key and call the run's tools through its socket. `by` names the command
  whose socket was used.
- Until its command ends, each open connection holds a file descriptor and up to 1 MiB in the
  engine's process, which a sandbox's limits don't cover. The workspace `timeout` bounds how long,
  and a command has at most 64 calls in flight at once, so it holds at most 64 file descriptors and
  64 requests of up to 1 MiB each in the engine's process. A script that opens more finds its
  connection closed, and should retry.
- A command makes at most 10,000 calls, sends at most 2 MiB of request lines and nests a call's
  `args` at most 128 levels deep. Past each, a request is answered with its error before it is
  parsed — `tools/bridge: this command made its 10000 calls; no more are answered`,
  `tools/bridge: this command sent its 2097152 bytes of calls; no more are answered`,
  `tools/bridge: a call's args nest deeper than 128 levels` — and leaves no step and no event. So
  one command adds at most 10,000 steps to the run's record, the engine parses at most 2 MiB of its
  requests however they are shaped, which bounds the arguments it holds and records, and every
  argument it records nests well inside the depth the record's serializer takes. A line counts
  toward the 2 MiB as it is read, and its key is checked only once it is parsed, so no line past
  the budget is parsed, one without the key included, while one without the key inside the budget
  is parsed and then refused. So a process that can reach the socket file can make the engine parse
  at most 2 MiB of lines per command — about a 350 ms stall of the event loop at worst, as
  measured — and with junk lines spend the rest of that command's budget, so the command's own
  calls are refused from then on, as it can hold its 64 slots. Keeping the socket to the command is
  the sandbox's job, as it is for the key.
- The socket lives only while the command runs; something the command leaves running cannot call
  afterwards.

A backend gets the bridge when it has a `root` attribute: a local directory its commands see at
the same path, as `Local`'s is, where each command's socket is created. A `root` that means
anything else gets a socket created at that path, so a backend without such a directory has no
`root`. `Workspace(..., bridge=False)` turns the bridge off, and so do a backend without a `root`
attribute and a run that offers no tool it could carry: the `Shell` description then does not
mention it. A command started with `background=true`, and a `Shell` that code dispatches outside a
run's own `Shell` call, get no socket either. A command without the bridge gets no `RFNRY_TOOLS`.
A root too deep for a socket's path — about
107 bytes on Linux, 103 on macOS — runs the command without it, and its `stderr` ends with the
reason: `tools/workspace: no tool bridge for this command: AF_UNIX path too long`.

Without a sandbox, the kill reaches every process the command started except one that leaves its
process group with `setsid`; the process namespace of a container or of bubblewrap closes that gap.
Such a process keeps running, but not on the command's pipes: once the command has exited and its
output is no longer read, `Local` closes them, so it holds none of the engine's files and what it
writes there after is lost.

A backend is three methods; the library does the rest — the tools, the time limits, the output
caps, stopping what a command leaves behind:

```python
class Backend(Protocol):
    async def spawn(self, command: str, env: Mapping[str, str]) -> Process: ...
    async def read(self, path: str, limit: int) -> bytes: ...
    async def write(self, path: str, data: bytes) -> None: ...
```

`spawn` starts a command in the workspace with no input, with `env` added to its environment, and
returns at once a `Process`: its `stdout` and `stderr`, read until they end, `wait()` for its exit
code, and `kill()` to end it and everything it started. Every kill the workspace starts runs to its
end, and the call that started it waits for it: a stop or a cancel of a `Shell` call, of
`Job(stop=true)` or of `close()` returns once the kill has returned — `close()` once the jobs it
stopped have ended too — however many cancels come meanwhile. A cancel that comes while `spawn` runs
waits for it to return, then for the kill of what it started — with `background=true`, for its job
to be listed, so `close()` stops it. So `spawn` and `kill()` must return promptly. A kill counts as
the first stop, so a second ends a command at once: the next cancel after the one that started a
kill — even one that comes while `spawn` still runs, or at the same moment — cancels that kill,
once, and so does the second cancel during a kill the workspace started on its own, for a command
past its timeout or that has ended, `Job(stop=true)` or `close()`. The call still waits for the
kill: a cancelled `kill()` must still end the command and everything it started, at once, before it
returns; `Local` sends SIGKILL in a `finally`, and a backend of yours must do the same. A `kill()`
that raises is the call's error; when the call is already leaving with a cancellation, or with the
error of a `wait()` that raised, it is logged as an error instead, `tools/workspace: could not stop
the command: …`, and the kill that follows a job's end puts its error in the job's output. `read`
refuses a file larger than `limit`; `write` creates parent directories. A path is the model's,
relative to the workspace, and what it can reach is the backend's to decide; an error a backend
raises reaches the model as the tool's error, a `TimeoutError` of its own as it is: only the
workspace's `timeout` makes a `Shell` return `timed_out` or a `Write` or `Edit` fail with
`tools/workspace: Write did not finish within 120s`. A backend's `description`, if it has one, is
ignored: the machine the model has is told in the agent's markdown, not by the workspace.

The library has no notion of a secret. A key a user types is the run's input; a key you put in
`Local(env=)` is the commands' environment, which a command can print. Either way the record keeps
what the model was given and what commands printed, exactly, in the run's scope and in what you
export from it. `record=False` keeps a run out of the record; `delete_runs` and `delete_scope` take
it out later.

An agent without `workspace: true` never sees these tools, even when the run has a workspace; an
agent with it, run without one, is a `ConfigurationError`, and so is a decision agent that declares
it. Delegated and routed agents receive the run's workspace on the same terms. `stop.now()` stops a
running command the same way the timeout does, and a second one kills it at once; either leaves jobs
running. You own the workspace and its lifetime: a temporary directory per run, or one kept for a
project. A workspace belongs to one tenant, the one its `scope` names: a run, a tool or a replay of
another scope refuses it before anything runs — `engine: workspace belongs to scope 'acme', not
'globex'` — so a workspace, its files and its jobs never cross tenants.

### Approving tool calls

By default an agent calls any tool it has. Pass `approve` and every tool call waits for your answer
first — the shell, a `Write`, a refund, a `Delegate`. It receives the agent's name and the call, and
returns `True` to run it or `False` to refuse:

```python
async def approve(agent, call):
    if call.name in ("Write", "Edit"):
        return True
    return await ask_a_person(agent, call)

run = await engine.run("coder", scope="acme", input="make the failing test pass",
                       workspace=workspace, approve=approve)
```

Where the person is — a thread, a dashboard, a chat app — is yours; the engine only waits. "Allow
everything" is no `approve`; "ask every time" is one that always asks; "allow edits, ask for the
rest" is the function above; "always allow this command" is a set the function keeps. A refused call
does not run: the model reads `NotApproved: Shell was not approved` and carries on, and a reason
reaches it through the `inbox`. Each step records `approved` — `True`, `False`, or `None` when
nobody was asked — and the exported rows keep it, so a person's refusals become labels. A
workspace's calls carry what a person needs to decide: `background` on `Shell`, `wait` and `stop` on
`Job`.

Time spent waiting is not a tool's time: the deadline starts when the call runs. `stop.now()` ends
a wait. Delegated and routed runs are asked through the same function, with their own agent's name;
`engine.tool()` and `EngineTools` take `approve=` for the runs they start. An `approve` that returns
anything but `True` or `False` is a `ConfigurationError`. The run waits in memory, so an approval
that takes hours is better refused now and continued later from `run.conversation`.

### Refining

The refinery records and exports what your agents did, with every label and its source. What you do
with it — evaluate, improve prompts and rules, train or distill models, audit — is yours.
[`evaluation.md`](evaluation.md) is a guide to doing it: capturing the right answer, the correct
label and what really happened, and the three levels of improvement they feed.

Attach a refinery and every run is recorded when it ends — one that raises after its first model
call too, as far as it got, with its error. A recording that fails is logged as a warning and
never fails the run, as one does when a model's tool call nests its `args` past 254 levels of lists
and objects, `args` itself the first, or fewer in a delegated or routed run, deeper than the
record's serializer goes (`ValueError: Circular reference detected (depth exceeded)`).
`refinery.run(scope, run_id)` reads a recorded run back, its delegated and routed runs in place, to
continue it anywhere — see [Running an agent](#running-an-agent). The record keeps a run's input as
it was given, and reads it back so: a generation's list input is the parts carried by the user
message the input became, the skill's part aside, when their JSON equals the stored input, and JSON
otherwise — a part-shaped class of your own is stored as it dumps itself; a decision's list state
holding media reads back as parts only when every item is written exactly as a part, all its fields
and no other, and as JSON otherwise.

```python
refinery = Refinery(PostgresStorage(pool))
await refinery.check()
engine = Engine(root, models={"strong": llm, "judge": dm}, refinery=refinery)
```

Labels come from three places. A person reviews a run — the button under a reply, the correction a
supervisor types, the right answer to a question that was decided wrong:

```python
await refinery.review("acme", run.run_id, Review(stance="objection", correction="Refund the second charge in full."))
await refinery.review("acme", decision.run_id, Review(correction={"intent": "technical"}))
```

The world reports whether the action worked:

```python
await refinery.outcome("acme", run.run_id, Outcome(status="failed", detail="Customer escalated."))
```

Each call stores a new label. An app that labels from events more than one process hears — a staff
reply every replica of a backend receives — or that retries after a crash passes `label_id=`, an id
derived from the event such as the message id:
`refinery.review("acme", run.run_id, review, label_id=message.id)`. The first call stores the label
under it; every later call with that id on the same run, racing or not, returns that label and
writes nothing, and one on another run is an `InputError`.

And models label what people did not. A **judge** is a decision agent in your tree whose questions
are about an episode; it sees the input, the reply, every tool call with its result — one a
command made through the [tool bridge](#workspace) with its `by` — and the error a failed run ended
with, with each image, audio or document replaced by a placeholder naming it —
`[image: image/jpeg]`, `[document: application/pdf, "policy.pdf"]` — while the recorded run and the
exported datasets keep the media. A judge sees neither the system prompt nor `run.preload`, so a
rule in `AGENT.md`, or one a preloaded agent never had to `Read`, is not in front of it; ask about
them in the question when they matter:

```
agents/judge/
  AGENT.md                   name: judge   kind: decision   model: judge
  questions/grounded.json    check   Is every claim in the reply supported by a tool result?
  questions/rule.json        check   Does the reply apply the refund rule as written?
  questions/done.json        score   levels: Not addressed, Partly, Fully, Fully and confirmed
```

A **teacher** is a generation agent that answers a decision agent's questions with a rationale, so
a decision made without ground truth still gets one; it reads each question as the run asked it,
structure written as compact JSON. Both run when you say so, over runs they have not labeled yet:

```python
labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
labeled.judged, labeled.taught, labeled.errors
```

A judge or teacher the tree does not have is a `NotFoundError`, and one of the wrong kind a
`ConfigurationError`, before any run is labeled. A run that could not be labeled — the model
failed, or the teacher answered a choice with no option, a score with no level index, a check with
anything but true or false — is named in `labeled.errors` with why, and stays unlabeled for the
next call. Every label carries its source. A
judge's score is exported as a judge's score, never as a person's. A judge's or a teacher's label also
names the `model` and `settings` of the run that made it, so a score can be traced to what produced
it; a person's and the world's carry `null` for both.

Export writes datasets to any object with a `put(key, bytes)`:

```python
class Folder:
    def __init__(self, root): self.root = Path(root)
    async def put(self, key, data):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        (self.root / key).write_bytes(data)

exported = await refinery.export("acme", Folder("datasets"), since=last_watermark)
exported.keys, exported.rows, exported.watermark
```

Keys are `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, so repeated exports add files and never
overwrite. A run in a mode goes under `{scope}/{agent}.{mode}/{dataset}/{timestamp}.jsonl`, since two
modes may hold one question id with different wording; each row carries its `mode`, and
`mode="chat"` exports only that mode. Keep one watermark per `mode` value, and never pass a filtered
export's watermark to an unfiltered one: the runs it did not cover, recorded before it, are skipped
with no error. An agent named like a mode of another is refused at load, so a key still names one
agent. `since` selects every run recorded or labeled after it, written in full with all its
labels. A run that raised — its `error` set — is in no dataset; replay and label it like any other.
A router whose routed run raised is the exception: it answered, so its decision rows are written.
A run is recorded when it ends, so a long run still working at one export is in the next, whenever
it started; the watermark covers both, so a review of a run you already exported reaches the next
export. Rows are keyed by `run_id`: when a run appears again, its latest row supersedes the earlier
ones — dedupe by `run_id`, keep the latest. Because of that you may pass a `since` a little before
the last watermark: a run is stamped just before it is written, and services recording with their
own clocks can disagree by a little.

Every row carries `"holdout": true|false` — the rows kept aside so any measurement on them is
honest. It is derived from a hash of the run id (`holdout_percent=20` by default), so a run's flag
never changes between exports; `is_holdout(run_id)` gives the same answer in your own code. Three
datasets:

| dataset | one row per | what it holds |
| --- | --- | --- |
| `generation` | generation run | skill, context, tags, model, settings, fingerprint, system, preload, start, messages, tools, conversation, steps, reply, stopped, stop_reason, usage, every label |
| `preference` | human correction | tags, fingerprint, model, settings, prompt (system, preload, messages), chosen (the correction), rejected (the reply) |
| `decision` | question of a decision run | tags, model, settings, fingerprint, state, question, answer, probabilities, confidence, truth, truth_source, truth_model, truth_settings, outcome |

Every row carries its run's `tags`, `{}` for a run that had none, and its `fingerprint`,
`{"agent": …, "tools": …, "questions": …}` — `questions` is `null` for a generation run; group or
filter the rows by them in your own code. Every row also carries, right after `run_id`, the run's
`parent_run_id` — the run that delegated or routed to it, `null` for a top-level run — and its
`call` as `Run.call` holds it (`{run_id, agent, scope, tool, call_id}`); a dataset regroups its rows
by tree with them.

`model` and `settings` are the model that wrote the run's reply, or made its decision, and the
settings it reported or its adapter declared, `null` when neither said; a preference row's name the
ones that wrote `rejected`. Each exported label carries the `model` and `settings` of the judge or
teacher behind it, and a decision row's `truth_model` and `truth_settings` are the teacher's when
`truth_source` is `"teacher"`, `null` when it is `"human"`.

The first two are a generation agent's episodes and the corrections people made to them. A
generation row's `tools` is what the model was offered — `{name, description, input_schema}` each —
`preload` the parts `preload:` sent ahead of `messages`, `[]` without it, and `conversation` is
`run.conversation` in order: history, input, assistant turns with their `tool_calls` (`{id, name,
args}`), tool results with `is_error`, the final reply. Together they fit standard fine-tuning and
eval formats with a few lines of mapping; the library writes no vendor's format. Parts are written
inline, their bytes as base64, so a row is self-contained; strip the media from a row you do not
want it in. The third is a decision agent's answers beside what turned out true: a calibrated model
is one whose 0.8 was right 80% of the time, and this is the table that says whether it was:

```json
{"run_id": "…", "parent_run_id": null, "call": null, "holdout": false, "agent": "frontdesk", "tags": {"arm": "b"}, "question_id": "intent", "answer": "billing",
 "probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05}, "confidence": 0.7,
 "truth": "technical", "truth_source": "human", "truth_model": null, "truth_settings": null,
 "outcome": "failed"}
```

`refinery.conflicts(scope)` lists runs that carry two different human corrections — a dataset can
hold either, not both, and the library reports rather than guesses.

Replay measures a candidate — a new folder, a new model, any `Engine` you built — on what your
recorded runs were given. Each pair holds the recorded run, its labels and the candidate's run:

```python
candidate = Engine(root, models={"strong": llm, "judge": dm})

async def read_only(agent, call):
    return call.name in {"Read", "Grep", "Glob", "LS", "Load", "Delegate", "lookup_order"}

async for pair in refinery.replay("acme", candidate, agent="support", holdout=True, labeled=True,
                                  approve=read_only):
    pair.recorded.reply, pair.labels, pair.candidate.reply if pair.candidate else pair.error
```

> **Replay runs the candidate's tools for real.** A tool that refunds refunds again, a tool that
> sends a message sends it, a `Shell` runs. Pass an `approve` that refuses every tool that changes
> anything — `read_only` above lets through the builtins that read the agent's folder, `Delegate`,
> whose runs it is asked about too, and a lookup — or replay with `tools=` stubs that only read: a
> run's tool wins over the folder's tool of the same name. A refused call reaches the model as
> `NotApproved`, as in any run.

It replays the top-level runs of `agent` in the scope, oldest first. `holdout=True` keeps the
held-out ones, by the same hash and `holdout_percent` as `export`; `labeled=True` those a person
reviewed or the world reported on; `tags=` those whose tags contain these, as `delete_runs` matches;
`fingerprint=` the runs one version produced — the `run.fingerprint` of any of them, its `agent`,
`tools` and `questions` all equal; `limit=` the first so many. Each is a run of the candidate,
as `candidate.run(agent, scope, …, record=False)` would be but never reading `/name`, with the
recorded input — parts as parts — the history sent before it, the
skill and the `context`, so the candidate sees what the recorded run saw, its own skill file
read for a recorded skill. A run invoked by `/name` recorded that skill and the rest of its input,
and the replay sends that pair: `/name` is never read again, so a rest that itself starts with
`/<skill>` replays as text, as does a recorded input of `/close x` whose `close` has become public
on the candidate since. A decision run replays its state, and asks its runtime questions again
as they were recorded while the questions of its folder come from the candidate's folder, its
scripts filling theirs anew — a runtime id the candidate's folder now asks fails that pair; when its
route reached a generation run, that run's input, history and context go too. What a record does not
keep is yours to pass again — `expect=`, `approve=`, and the `tools=` and `workspace=` the runs were
given. Replays given one workspace share its files and its jobs, and a job outlives the replay that
started it: close the workspace when the replays are done, and the candidate, which stops its
workers. The candidate delegates and routes as it does, so a run with children replays as one run.

`repeat=3` replays each run three times: each attempt is a run of its own and a pair of its own,
with its own copy of the recorded run and its labels, so what you change in one pair reaches no
other attempt; `pair.attempt` counts them from 0, and the three come one after another; `limit=` still counts
recorded runs. A model samples, so the same candidate on the same input can answer differently, and
the attempts show how much. `concurrency=4` keeps up to four replays going, attempts included, and
while you read a pair at most three more run; with the default of one, a replay starts when you ask
for its pair. Pairs come in recorded order either way. A candidate that fails yields its pair with
`error` — `"ModelError: …"` — and `candidate=None`, and the replay goes on, the run's other
attempts included. Nothing is recorded, no label is written, and nothing is graded: the pair holds
both runs with their models and fingerprints, and what you compare is yours.
[`evaluation.md`](evaluation.md) grades the replies with a decision agent.

### Agents kept in a repository

The engine reads a folder; a new version of the agents is a new `Engine`. Keep the tree in a git
repository — people who do not write code edit it in the host's web UI, and every change is a
reviewed pull request — check each commit out into a folder of its own, build an engine from it, and
swap which engine new runs use. A run in flight finishes on the engine it started with. Rolling back
is loading an older commit, and since a commit's folder never changes, a half-written checkout is
never live. What an agent's prompt holds — `AGENT.md`, the map, the list of its skills, the preload,
an invoked skill, its tools' definitions and scripts — is what the engine loaded at `Engine()`; what
the model reads with a tool, and the data a script opens, is read from disk when it asks, so remove
an old folder only once its runs have finished. Each engine starts the workers of its own tree's
tools, which run the scripts as it read them, so close the engine you swap out — `await
engine.close()` — once its runs have finished too. A root that is itself an agent is named by its
own `name:`, so every commit keeps one name whatever folder it lands in. The checkout is yours —
`git worktree add`, a clone, an archive:

```python
def load(commit):
    return commit, Engine(checkout(commit), models={"strong": llm, "judge": dm}, refinery=refinery)

live = load("4f2c9e1")
commit, engine = live
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit})

_, retired = live
live = load("a81d07b")
await retired.close()

arms = {"a": load("4f2c9e1"), "b": load("a81d07b")}
arm = "ab"[hashlib.sha256(b"user-42").digest()[0] % 2]
commit, engine = arms[arm]
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit, "arm": arm})
```

Tag each run with the commit it came from and the record, its delegated and routed runs, and every
exported row say which version of the tree answered. The tag names a version; the fingerprint proves
it. A tag says what you meant to load, and a checkout with an uncommitted edit carries the same tag;
`run.fingerprint.agent` is taken from the files the engine read, so two runs are the same agent when
their fingerprints match, whatever their tags say. The same tag is an A/B test: load two commits
side by side, send each user to one by a stable hash of their id, tag the run with its arm, and
group the exported rows by `tags["arm"]` to compare the labels and outcomes of whole versions of the
tree.

### Running many at once

One `Engine` per process serves every run at once; scale out with processes. A run in flight is tied
to the process that started it, so route a chat's later messages, its stop and its inbox to that
process. The engine's folder tools run in its own workers beside that process — one process for the
whole tree by default, and as many for a group of agents as [`workers=`](#tools) gives it — each
process taking 32 plain `def run` calls at once on its threads, and `async def run` calls without a
limit on its event loop.

The tools' file work (`Read`, `Grep`, `Glob`, `LS`, `Load`, the history search, and `Local`'s file reads and
writes; its commands are processes of their own) runs on the event loop's default executor,
`min(32, cpus + 4)` threads. When many runs search at once, size it with
`loop.set_default_executor(ThreadPoolExecutor(64))`. A turn that calls no file tool never waits on
it.

For an agent with `Read` or `Load`, the FILES map goes out with every model call, roughly 6 tokens
per file. A corpus of thousands of files belongs behind a retrieval tool, not in an agent's folder.

With reasoning on, a run's CPU grows with the history it is sent. The window is yours; see
[Long conversations](#long-conversations).

## Tools

One concept, one word. A **tool** is a name, a description, an input schema and what runs when the
model calls it. A **`Tools`** is a set of them: what a model receives, and what your code can call
the same way.

```python
class Tools:
    def definitions(self) -> list[dict]                 # [{name, description, input_schema}]
    async def dispatch(self, name: str, args: dict) -> str | dict | list[Part]
    def __add__(self, other) -> Tools                    # union; a duplicate name is refused
    def only(self, *names) -> Tools
    def without(self, *names) -> Tools
    names: list[str]                                     # every tool's name, in order
```

An agent's tools live in its tree, beside its instructions. A tool is two files in a folder named
`tools/`: `<name>.json`, what the model reads about it, and `<name>.py`, what runs. The tool's name
is the file's stem. An agent sees the tools of its own `tools/` and of the `tools/` of every folder
above it, up to the root, and the nearest wins a name clash:

```
agents/
  tools/notify.json, notify.py                 every agent under agents/ sees notify
  customer_support/
    tools/lookup_order.json, lookup_order.py   every agent under customer_support/ sees lookup_order
    tools/_client.py                           a helper of the scripts beside it, not a tool
    triage/AGENT.md
    billing/
      AGENT.md                                 tools: [lookup_order, refund, Read]
      tools/refund.json, refund.py             only billing, and the agents inside it, see refund
```

A `tools/` folder on an agent's path from the root — the root's, a group folder's or the agent's own
— is a declaration, as an agent's own `questions/` and `skills/` are: it stays out of every map and
of any folder `preload:` names, since the model sees its tools as tools, and goes ahead only when
`preload:` names it directly (`'./tools'`); it stays readable to the tools of the agent whose folder
holds it, and counts in the fingerprint of every agent that sees it; a folder named `tools` anywhere
else is content like any other. Two `tools/` folders on an agent's path may each hold a tool of one
name: the agent gets the one nearer it, its own folder's before a group's and a group's before the
root's, as a run's own `tools=` replace both. A decision agent has no tools, and a `tools/` folder
on its path is not offered to it. A `.json` without its `.py`, or a `.py` without its `.json`, is a
`ConfigurationError` naming the file; a file whose name starts with `_` or `.` is not a tool, and
`_name.py` is a helper the scripts of its folder, and of every `tools/` folder below it, import. A
helper named like a module of the standard library, `_stat.py`, `_weakrefset.py` or `_strptime.py`,
is a `ConfigurationError` too, since the worker can't load both: `tools/folder: <path> is named like
the standard-library module '_stat', and the worker can't load both; rename it`. A question's
helper, `questions/_name.py`, is refused the same way.

`<name>.json` takes three keys:

```json
{
  "description": "Refund one charge of the customer's. Returns the refund's id and status.",
  "input_schema": {
    "type": "object",
    "properties": {
      "charge_id": {"type": "string", "description": "The charge to refund, such as ch_81"}
    },
    "required": ["charge_id"],
    "additionalProperties": false
  },
  "output_schema": {"type": "object", "required": ["refund_id", "status"]}
}
```

`description` is what the model reads about the tool — what it does and what it returns — and its
first line is what the prompt's list of tools shows. `input_schema` is a JSON Schema of the
arguments, its descriptions included, whose top level is `"type": "object"`; `output_schema`,
optional, is one of the result. When the tree loads, each file is checked and both schemas are
compiled — any draft, 2020-12 included, and never a reference fetched from the network — and an
unknown key, a missing or blank description, an `input_schema` that is not an object or a schema
that does not compile is a `ConfigurationError` naming the file: `tools/folder: <path> has the unknown key
'inputSchema'; a tool's JSON takes description, input_schema and output_schema`. The model is
offered the name, the description and the `input_schema`; `output_schema` never reaches it. A call's
arguments are checked against `input_schema` before they reach the script, and its result against
`output_schema` after; either that does not fit is a tool error the model reads, `InputError:
tools/declared: invalid arguments for refund: 5 is not of type "string" at charge_id`, so the script never
sees arguments its schema refuses and the model never reasons over a malformed result.

`<name>.py` holds the logic: a function `run(args, call)`, plain or `async`.

```python
import _client


async def run(args, call):
    refund = await _client.refund(call["scope"], args["charge_id"], reference=call.get("run_id"))
    return {"refund_id": refund.id, "status": refund.status}
```

`args` is what the model wrote, already checked. `call` is the call it serves, set by the engine,
never by the model: its `scope` — the tenant, always — its `tags`, and inside a run its `run_id`,
`agent`, `tool`, `call_id` and `by`, the fields of `CallInfo`, `by` naming the `Shell` call whose
command made the call through the [tool bridge](#workspace), `None` for the model's. `tags` are
the tags of the run the call serves, which a delegated or routed run inherits, and outside a run
the ones given to `EngineTools`, `{}` without. What is a tenant's travels in `call`, and so does
what the run is about: a tool bound to the ticket or the customer a run serves reads it from
`call["tags"]`, which your code set and the model can neither see nor change, never from an
argument the model writes. What every tenant shares, a URL or a token, comes from the environment,
which a worker inherits from your server as it is — the engine never reads it. `run` returns a string; JSON — an object, a list, a
number — the model reads as JSON; or a list of parts, each an object with the `type` and the fields
of a `Text`, `Image`, `Audio` or `Document`, media `data` in base64, which reach the model as parts:
a PDF the model reads itself, an image it looks at. A result JSON cannot hold, a `datetime` or a
`set`, is a tool error naming its type; arguments it cannot hold, passed from code, are an
`InputError` before any worker sees them (`workers/pool: the arguments for refund are not JSON: Object of
type datetime is not JSON serializable`). A script imports the standard library, installed packages
and the helpers of its own `tools/` folder and of each `tools/` folder above it on the way to the
root — their `_name.py` files, by name, at the top of the script, a helper of a nearer folder
replacing one of the same name above it, as a tool does, for the helpers it imports in turn too.
One agent's version of a shared tool sits in its own `tools/` under the shared tool's name and
imports the shared tool's helper, with no second copy of the code:

```
agents/
  tools/_read.py, order_history.json, order_history.py     every agent: the whole history
  frontline/tools/order_history.json, order_history.py     frontline: its own, importing _read
```

A tool's script or a package beside it, or in a `tools/` folder above it, is no helper, and
importing one is a `ConfigurationError` at load (`tools/folder: <path> imports 'b', which sits in its
tools/ folder but is not a helper; a tool reaches only the helpers of its tools/ folder and of those
above it, so move the shared code into a helper _b.py`, or `which sits in the tools/ folder <folder>
above it`), unless the name is one of the standard library's, `time` or `json`, or resolves to an
installed package. A helper of a `tools/` folder beside the script's, or below it, is out of its
reach. Every agent that sees a `tools/` folder sees the ones above it, so its fingerprint covers
every helper its tools can import.
It never imports the engine, nor your server's code. When the tree loads, each script is read
without running it: a script with no `def run` or `async def run` at its top level is a
`ConfigurationError` (`tools/folder: <path> has no run at its top level; a tool defines def run(args,
call) or async def run(args, call)`), and so is an import of `rfnry_engine`, of a module whose file
lies in the project that holds the tree — the nearest folder at or above the root with a
`pyproject.toml` — outside the standard library, the installed packages and the tree's `tools/` and
`questions/` folders, of a module relatively, or of a helper inside a function or class (`import it
at the top of the script, since a helper is reachable only while its folder's scripts load`), each
naming the file and the module. The installed packages are those of the site folders of your
server's interpreter and of the folders the interpreter that runs the workers passes to
`site.addsitedir` as it starts — its own `site-packages`, the user's, and every site folder a `.pth`
file or `sitecustomize` adds, as `uv run --with` does with your project's `.venv`. A folder a `.pth`
line merely puts on the path, as an editable install of your own app does, stays your server's
code. To learn those folders, loading runs that interpreter once, with `-P -S` and for at most ten
seconds, and only when a script imports a module of the project that your server's own site
folders do not hold; that run executes your `.pth` lines and `sitecustomize` as any start of it
does, and when it cannot answer, the server's own site folders are the installed packages. A server
package named `tools` is your server's code like any other: what counts is the folder the module's
file lies in, not its name. That catches a mistake; it is not a sandbox, and a script is trusted as
your server's code is.

A tool runs in a worker: a process the engine starts with your server's interpreter, so it has the
packages your server has, which runs the scripts of its group's agents — their tools' and their
questions' ([Decision agents](#decision-agents)) — and serves their calls. Workers run on POSIX
systems only, Linux and macOS, as `Local` does: each is a session of its own, stopped with every
process it started by signalling its process group. The scripts and their
helpers are read when the tree loads, as skills are — the text the import check read and
the fingerprint covers — and a worker runs that text, under each file's own path, so `__file__`
still finds the data beside it: an edit to a tool reaches the next `Engine`, never the one running,
whose records keep naming what ran, while a data file a script opens when it is called is read
from disk then. A worker starts at the first call that needs it, never when the engine is built,
and again after a crash, and lives until you close the engine — `await engine.close()`, or leaving
`async with Engine(...) as engine:` — which stops every worker and whatever it started, and raises
the first error stopping one raised once all are stopped, logging the others; it waits for a worker
a call is still starting, to stop it too, and a cancel of `close()` reaches you once the workers on
your loop are stopped — one still starting is stopped as it comes up — and those on another loop are
handed to it, without waiting for that loop. The engine watches each worker with one thread of its
own, which waits on the worker's exit and ends with it, so a server counting its threads counts one
more for each worker running. A closed
engine stays closed: a run, a tool call or an `EngineTools` after `close()` is an `InputError`,
`engine: this engine was closed; build a new Engine`, before any model is called, so nothing is
recorded of it, and a run still in flight when it closed starts no other run — its `Delegate` call
reads that error, its route fails with it. A plain `def run` runs on one of the worker's 32 threads,
and calls past those wait; an `async def run` runs on its event loop, as many at once as come. The
worker goes by the `run` it loaded, whatever its file spells: a `run` a decorator made a coroutine
function runs on the event loop, any other on a thread, and what that one returns, if it can be
awaited — a coroutine, from a plain decorator around an `async def` — is awaited on the event loop.
What a script prints goes to the log, at DEBUG on the `rfnry_engine.workers.pool` logger, never into
its result: a record per line, and 64 KiB at a time of a line whose end has not come; with DEBUG off
it is dropped as it comes. A script that raises, `sys.exit()` included, is a tool error the model
reads, `WorkerError: ValueError: no order A-1`, and the worker goes on. What a script returns or
raises crosses back as JSON of at most `Engine(worker_result_bytes=)` bytes as the worker writes it,
a non-ASCII character escaped to six, 32 MiB unless you set another number, `None` for no limit;
past it the worker sends a short error in its place, so a result too large never reaches your
server, the model reads `WorkerError: workers/process: refund sent back 40000023 bytes, over the
limit of 33554432 bytes`, and the worker goes on. A result under the limit still costs your server
about three times its size while it is read, for each worker with one in flight, and recording the
run holds its body, which stores a result twice, in its step and in the conversation, at most twice
over while it is written. A worker that exits — `os._exit()`, a signal, out of memory, a crash in a
C extension — fails every call it had in flight, even while a process it forked still holds its
pipes: `WorkerError: workers/pool: the worker for refund stopped: exited with code 3`, or `killed by
signal 9` when a signal ended it, the run goes on, and the next call starts a new worker. A worker
that sends back a line that is not a reply, as a process it forked writing to its pipe could, is
stopped the same way, its calls failing with `workers/pool: the worker for refund stopped: it sent a
line that is not a reply`. Past `tool_timeout` the model reads `TimeoutError: workers/pool: refund
timed out after 30s`, a deadline that counts the wait for a worker too: an `async def run` is
cancelled in the worker and holds none of its threads, unless it blocks the worker's event loop —
`time.sleep` or blocking I/O inside it — so that its cancel is still unanswered when a call sent
after it passes its deadline too, and then each of those calls holds a thread; a `def run` cannot be
stopped, so its thread runs on and its result is discarded, and once every thread of a worker is
held so, the next call stops that worker and starts a new one, and the calls the old one still had
in flight fail with `WorkerError: workers/pool: the worker for refund stopped: every thread was held
by a call past its deadline`.

An engine serves one event loop at a time, as a workspace's jobs live in one loop, and its workers
follow the loop that calls them: a call from a loop other than the one that started them stops
those workers and starts fresh ones on its own. A sync server keeps one loop running in a thread of
its own and hands each run to it, `asyncio.run_coroutine_threadsafe(engine.run(...), loop)`.
`asyncio.run` per request or per task, or a test's own loop, starts the workers anew each time, and
loops in several threads that call one engine at once keep stopping each other's workers, so give
each live loop its own engine. A worker that stops — for another loop, a retirement, a crash or
`close()` — is reaped and its pipes closed then and there, so a thousand `asyncio.run` calls leave
the engine holding no more files or processes than one does. `close()` from another loop stops the
workers at once: while their loop runs in another thread, the stop runs on that loop — or in the
caller, should that loop stop before it gets to it — and `close()` returns once it is done. A
worker that loop was still starting when it stopped is killed and reaped in the caller, while its
pipes wait for that loop: they close when it runs again. Close the engine before its loop ends: closing it later from another loop still stops every worker, but
Python warns of the worker pipes left on the closed loop (`ResourceWarning`).

By default every agent of the tree shares one worker. `Engine(root, models=...,
workers={"customer_support": 1, "reports": 4})` makes each folder named — by its path from the root, such as
`customer_support` — a group of its own, for every agent inside it at any depth, with that many
processes, and a call goes to the group's process with the fewest calls in flight; the agents
outside the named folders share the default group, one process. A name that is not a folder holding
an agent, at any depth below it or itself, or a folder inside another one named, is a
`ConfigurationError`:
`engine: workers names 'support/billing' inside 'support', which it names too; keep one`.

Every tool an agent gets is on: the tools of its folders, the file tools, and the tools of the run,
given to its own run with `run(tools=)` or to the run that delegated or routed to it, below.
`tools:` in its `AGENT.md` keeps exactly the ones it lists — `tools: [lookup_order, Read]` keeps
only these, `tools: none` none of them, leaving the agent only the tools that come with their own
declarations, and an agent with `tools: all` or without `tools:` keeps every one — so an agent leaves out a tool it
inherits, from a folder above or from the run that delegated to it, by listing the ones it keeps.
In the tree above, billing keeps `lookup_order` from `customer_support/`, its own `refund` and
`Read`, leaving out `notify`, the other file tools and any tool a run would give it:

```markdown
---
name: billing
kind: generation
model: strong
tools: [lookup_order, refund, Read]
---
Refund a duplicate charge with refund. Never promise a refund the rules do not allow.
```

`Read Grep Glob LS Load` come with every generation agent, `Delegate` with `agents:`, a workspace's
tools with `workspace: true`, `History` with a run whose `start=` is past the beginning of its
history, and `Output` with `expect=`. A list keeps from the folders' tools, those five file tools
and the run's tools: each of the others comes with its own declaration, or with the run, and stays
as long as that does, so naming one in `tools:` is a `ConfigurationError` when the tree loads —
`tree: agent 'support' keeps 'Delegate' in tools:, which comes with its own declaration and stays
whatever tools: lists; leave it out of tools:` — and `Output` cannot be named in `tools:` either.
Any other name loads, since it may name a tool a run will give, which the tree cannot know, and is
checked when a run starts: a kept name that is neither a tool of the agent's folders, a file tool
nor a tool of the run is a `ConfigurationError` before any model is called, naming the ones the run
offers — `tools/engine_tools: agent 'support' keeps 'refnd' in tools:, which this run does not
offer; its tools are Glob, Grep, LS, Load, Read, refund` — which the model of a run that delegated
to it reads as the `Delegate` call's error, while a route raises it. A tool in a folder named
like one the engine gives the agent is a `ConfigurationError` naming its file, and `History`'s name
is taken on every run, so a tool named `History` is refused from the first.

Tools that must live in your process ride a run: `run(tools=)` takes a `Tools`, or anything with its
two verbs, for that one run — a tool bound to a tenant's connection, one that streams to a UI,
rfnry-memory's `MemoryTools`; a tool every run needs is the same `Tools` passed on every run. Each
is a `Tool`: its name, its description, its arguments and the async function that runs it, in your
process:

```python
class InvoiceArgs(BaseModel):
    """Which invoice to fetch."""

    invoice_id: str = Field(description="The invoice id, such as INV-204")

async def fetch_invoice(args):
    data = await invoices.get(args["invoice_id"])
    return [Document(media_type="application/pdf", name=f"{args['invoice_id']}.pdf", data=data)]

invoicing = Tools([Tool("fetch_invoice", "Fetch one of the customer's invoices, as its PDF",
                        body=fetch_invoice, args=InvoiceArgs)])
run = await engine.run("support", scope="acme", input=message, tools=invoicing)
```

The `description` is what the model reads about the tool and its first line is what the prompt's
list of tools shows. `args` is a pydantic model: its `Field(description=...)`s and its docstring
reach the model in the tool's schema, and it checks a call's arguments before the body runs. An
`input_schema` states the schema as a dict instead; a tool given only `input_schema`, and a `Tools`
of your own, receive the arguments as the model sent them. A body returns what a script does: a
string, a dict the model reads as JSON, or a list of parts. A run's tool wins over a folder tool of
the same name, since the run is nearer than the agent's own folder: that is how a test stubs a
folder tool that calls a real system, and how you swap in a version bound to a tenant's connection.
Two tools of one run with one name are refused, and so is a run's tool named like one the engine
gives the agent.

A run's tools ride the runs it delegates and routes to, as its tags do, and each agent keeps of
them what its own `tools:` lists. A lead given rfnry-memory's `MemoryTools` that delegates to a
writer with `tools: [Read]` keeps the memory tools for itself, and the writer is offered `Read`
alone; a writer without `tools:` gets the memory tools too, and one that needs one of them names
it, `tools: [Read, memory_search]`.

Every tool call inside a run has a deadline, `Engine(tool_timeout=30.0)` by default and `None` to
turn it off. When it passes, the model gets the error
`TimeoutError: engine: lookup_order timed out after 30s` and the run continues. A folder tool's
deadline is its worker's, above. `History` reads your storage under the same deadline.
A run's tools are async and the deadline cancels them, so a body that runs blocking work in a thread
must put its own limit on that work: cancelling the await does not stop the thread. A workspace's
`Shell`, `Job`, `Write` and `Edit` keep the workspace's own `timeout` instead. `Delegate` has no
deadline. The child run it starts is bounded by its own tool calls and `max_steps`.

The builtins are bounded whatever the deadline. `Read` returns UTF-8 text and refuses anything else
with a pointer to `Load`; `offset` and `limit` select lines. `Load` takes any file up to 20 MB and
hands it over as an image, audio or a document part, whichever its type says. The type comes from
the name of the file it opens, so a link goes by its target's name (by its own where the system
cannot say which file it opened, as on macOS), and for PNG, JPEG, GIF, WebP and PDF the first bytes
must agree — a `.png` that is not one is refused rather than sent — while a file with no known name
is typed by its bytes or sent as `application/octet-stream`. What the model can take is the
adapter's `media_types`, and past that the vendor's. `Grep` searches text only and skips the rest;
`Glob` and `LS` list every file. Neither `Read` nor `Grep` reads a file over 5 MB. `Grep` stops
after scanning 50 MB and notes the cut. A search that runs past 10 s in one call is an `InputError`
the model can act on. `Glob` and `Grep` walk at most 100,000 entries and note the cut. A tool that
reads the agent's directory itself resolves paths with `Jail`, which refuses one that leaves the
directory, as the builtins do. `Jail.open` opens a file the way the builtins and `Local` do: from a
place it checks is still the folder it was given, one folder at a time without following a link, so
a link swapped in after the check is refused.

A run's tool learns the call it serves with `current_call()`, a `CallInfo` — the run's id, its
agent, scope and tags, the tool and the call's id, what a script reads in `call` — so a side effect
can carry the run that caused it, as an audit row or a header does:

```python
async def refund(args):
    call = current_call()                                  # None outside a run
    await payments.refund(args["charge_id"], reference=call.run_id if call else None)
    return {"refunded": args["charge_id"]}
```

The run's `approve` and its handlers for the call see it too, and so do tasks the body starts, after
the call ends. A delegated run's calls, and those of a run an `engine.tool` call starts, name their
own run, while the rest of such a run — its model calls, its text handlers — runs inside the
caller's call and sees that one; `None` outside a run's tool call.

Tools that belong to a tenant go on the call that names the tenant:

```python
run = await engine.run("support", scope="acme", input=text,
                       tools=MemoryTools(memory, corpus_id="acme"))
```

That is how rfnry-memory mounts: its `MemoryTools` is a `Tools`, and neither library imports the
other. The set an agent runs with is itself a first-class object, so you can drive it without a
model — from a test, or from a loop of your own. `EngineTools` holds exactly what the model of such
a run is offered: the file tools, `History`, the folder tools, whose calls go to their worker under
`tool_timeout` with `call` holding the scope and the `tags` given to it, the workspace's,
`Delegate` and the run's `tools`.
`tools:` keeps of the folders' tools, those five file tools and the `tools` given only the ones it
lists; the rest stay:

```python
surface = EngineTools(engine, agent="support", scope="acme")
surface.definitions()                                        # what the model would see
await surface.dispatch("Read", {"path": "knowledge/refunds.md"})
await surface.dispatch("Delegate", {"agent": "triage", "input": "It is down"})

combined = MemoryTools(memory, corpus_id="acme") + surface.only("Read", "Grep")
```

A `Delegate` dispatched this way runs the child as its own top-level run, with the scope, `tools=`,
`workspace=` and `tags=` given to `EngineTools` and the `questions` it is called with —
what a `Delegate` inside a run would give it, its input likewise never read for `/name`. A refusal
raises here, where inside a run the model reads it.

An agent is itself a tool. `engine.tool(agent, scope=...)` gives any runtime — a live voice model, a
loop of your own — one tool that runs the agent: named for the agent and described by the
`description:` of its `AGENT.md`, which it needs, taking `{"input": str}` — and, as `Delegate` does,
`questions` when it is a decision agent — returning a generation agent's reply or a decision agent's
answers as a dict:

```python
consult = engine.tool("billing", scope="acme", name="consult",
                      context="Acme is on the enterprise plan.")
reply = await consult.dispatch("consult", {"input": "Was I charged twice?"})
triaged = await engine.tool("triage", scope="acme").dispatch("triage", {"input": "It is down"})
triaged["answers"]["urgent"]["probability"]                # 0.91
surface = consult + MemoryTools(memory, corpus_id="acme")
```

The tool's `input` is never read for `/name` and the run it starts takes no skill: the caller is a
runtime, not a person typing. A generation agent's tool returns its reply. A decision agent's
returns `{"answers": {…}}`, and when
it routed, also `"routed_to": {"agent": …, "reply": …}` — the routed run's result, nested the same
way when the target is itself a decision agent — so a caller that consults a routing agent gets the
answer it routed to, not only the answers.

It is a `Tools` of one, so it adds to any other. Each call is a top-level run under the scope it was
made with, recorded like any run — but not when the run that makes the call was started with
`record=False`, as a replay's is; `context=`, `tools=`, `workspace=` and `tags=` ride every call.
Mounted inside another run with `run(tools=)` it is a tool like any other, under `tool_timeout`, and
its run is not the caller's child — to nest with lineage, name the agent in `agents:`. As across
`Delegate`, `max_depth` counts the whole stack across it; a call through it starts a new input, so
the route chain resets there: an agent may reach a fresh run of itself through it, and only a route
into an agent already in the same chain of routes is refused.

Called inside a run, a consult is part of that run's foreground, as a delegated run is: its run
takes the caller's `stop` and `on_event`. The caller's `stop.now()` reaches the consult instead of
cancelling its call, so the consult ends with its record — one a script's call started included, as
"Calling tools from code" says — and whatever else the call was doing is cut once the consult has
taken the stop — a stop that lands before a call returns cuts it, even when its consult had just
finished, and that consult keeps its record — and the caller's step reads `stopped by the user`, as
a stopped `Delegate` step does; `after_step()` ends it after its current step. A consult a tool of
yours starts in a task of its own — two at once, say — is in the foreground too; one a script's call
started through the workspace's tool bridge ends with the script's command instead, unless
`stop.now()` ended the command, as "Calling tools from code" says. Its events reach the caller's
handler, told apart by `agent` and `run_id`:

```python
async def status(event):
    if isinstance(event, ToolCallStarted):
        print(event.agent, "is using", event.tool)          # billing is using lookup_order

consult, stop = engine.tool("billing", scope="acme", name="consult"), Stop()
run = await engine.run("support", scope="acme", input=message, tools=consult,
                       on_event=status, stop=stop)
```

A stop flows down, never up: a consult that is stopped ends alone, and its caller reads `stopped by
the user` and carries on. Outside a run — a live voice model, a loop of your own — pass the hooks
yourself. `on_event` rides every call; `stop` is a function called at each call that returns the
`Stop` of the turn the call belongs to, since a `Stop` belongs to one run and the tool lives as long
as the session:

```python
turn = {"stop": Stop()}

async def say_progress(event):
    if isinstance(event, ToolCallStarted):
        await voice.say("Checking that now.")

consult = engine.tool("billing", scope="acme", on_event=say_progress, stop=lambda: turn["stop"])

def barge_in():
    turn["stop"].now()                  # stops the consult running now; it is recorded, stopped
    turn["stop"] = Stop()               # the next turn's consults run normally
```

Given inside a run, these win over the caller's: a `stop` of your own, or `None` from the function,
leaves the consult out of the caller's foreground, and the caller's stop cancels its call as it
cancels any tool's. Outside a run, a stopped consult's `dispatch` returns what its run produced.
`EngineTools` takes the same two for the `Delegate` it dispatches outside a run.

## Storage

The refinery is the only part of the library that stores anything, and it stores in Postgres:
`PostgresStorage(pool)`, over an asyncpg pool you built, in a schema of its own. It sends and
reads every JSON value as text, cast in its SQL, so it works with any `json` or `jsonb` codec on
that pool, such as `json.dumps` and `json.loads` set for both, which never sees what it stores.
Each run's body reaches Postgres as a `bytea` argument, an object with the buffer protocol that
hands its JSON over and keeps no copy, encoding it again from the body each further time it is
asked, so the write holds it at most twice and a driver that asks twice sends the same JSON both
times; a `bytea` codec of your own on that pool must take any such object, as `bytes(value)` does,
and one that takes only `bytes`, such as a text codec calling `value.hex()`, fails every record
write with `StorageError`. A Postgres to try it on is one container, with engine's schema prepared
by the command the package installs:

```bash
docker run -d -p 127.0.0.1:5432:5432 -e POSTGRES_PASSWORD=postgres postgres:18-alpine
rfnry-engine create --url postgresql://postgres:postgres@127.0.0.1:5432/postgres
```

### Preparing the database

Your app never creates or changes engine's tables. `rfnry-engine`, the command the package
installs, prepares the schema before the app runs:

```sh
rfnry-engine create --schema engine          # the schema and its tables
rfnry-engine create --schema engine --sql    # print every statement, run nothing
```

`create` makes `engine_runs` and `engine_labels` and their indexes — among them `(scope,
parent_run_id)`, which a run's tree is found by, and a GIN index on the `tags` column of
`engine_runs`, which tag selection uses — and `engine_migrations`, its record of the migrations run.
While tables `initialize()` made sit in a shared schema such as `public`, `create` refuses and names
the `upgrade --from` that keeps their runs; `create --fresh` starts empty anyway. It takes the
connection string with `--url`, or reads Postgres' standard `PG*` variables, and never prints it.
`--sql` prints the statements as the command runs them — the `search_path` they run under, then one
transaction holding the migration lock — so they land in the same schema when you run them yourself.
Engine's tables never go in `public`. An app that connects as a role of its own, not the one that
ran the command, needs `USAGE` on the schema, `SELECT`, `INSERT`, `UPDATE` and `DELETE` on
`engine_runs` and `engine_labels`, and `SELECT` on `engine_migrations`.

Your pool points at the schema, and the app checks it at start:

```python
pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "engine"})
await Refinery(PostgresStorage(pool)).check()
```

`check()` only reads. It raises a `ConfigurationError` saying how the schema does not fit this
version and the command that fixes it: no engine schema on the pool's `search_path`, a schema behind
or ahead of this version, tables an earlier version's `initialize()` made, or a schema or tables the
pool's role cannot use. When the pool's first schema is another library's, it points at engine's own
schema instead of offering that one.

A version of engine that changes its tables ships a migration, and the schema follows it in place,
its data kept:

```sh
rfnry-engine upgrade --schema engine --sql    # what would run
rfnry-engine upgrade --schema engine          # lists the steps, asks, runs them in one transaction
rfnry-engine check --schema engine            # exits 0 when the schema fits this version
```

Take a snapshot and stop the app instances running the old version first: `upgrade` reminds you,
asks before it runs unless `--yes`, and runs every pending step in one transaction under a lock.

Every run and label a released version recorded reads back in this one: the upgrade to schema
version 3 rewrites each stored body in the shape this version records, whatever release wrote it —
a generation run from before 0.1.4 gets its `sent` in place of the `messages` it held, a key added
since, such as a step's `by` or a run's `settings`, its value for nothing, and a run's `call` the
tags of the run that made it — so `refinery.run`, `export` and `label` read every row, and a body
already in this shape is not touched. A body no release recorded — one written by hand, or cut
short — stops the upgrade before anything changes, naming it: `refinery/migrations: run 'r9' of
scope 'acme' holds a body no release of rfnry-engine recorded (KeyError: 'messages'); delete it
or restore its body, then upgrade again`. The rewrite reads every row once, in pages of a thousand.
The upgrade to schema version 4 drops `engine_runs.task` and rewrites each stored body without its
`task`, the same way: a run records no task. The upgrade to schema version 5 adds the nullable
`engine_runs.mode`; a run recorded before it has no mode.

Tables made by `initialize()` in rfnry-engine 0.1.4 or earlier are adopted where they are:
`rfnry-engine upgrade --schema <their schema>` records them, and gives a table from before 0.1.4 its
`tags` column and the indexes it lacked. It fills `tags` again from each recorded run's own tags, in
a table 0.1.4 made too: before, a character written as an escape starting `\ud` — Hangul, emoji —
was stored rewritten, so two different tags could select each other's runs. A tag holding a NUL or
an unpaired surrogate, which Postgres cannot store, is kept and matched with U+FFFD in its place.
Tables in a shared schema such as `public`, where a pool with the default `search_path` put them,
move into engine's schema in the same transaction:

```sh
rfnry-engine upgrade --schema engine --from public
```

and the pool's `search_path` then names `engine`. Run again, it has nothing left to do; until it
runs, `check()` and the command point at it whenever engine's schema is missing or empty while
tables `initialize()` made sit in another.

A run's row keeps `at`, when the run started, and `recorded`, when the refinery stored it — one
stamp for a run and every child run under it — and `runs(since=)` reads `recorded`. The tables are
prefixed `engine_`, and each rfnry library keeps a schema of its own, so one database holds
rfnry-memory's and rfnry-thread's beside them, each through a pool that names its schema:

```python
memory_pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "memory, public"})
engine_pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "engine"})
memory = Memory(rfnry_memory.PostgresStorage(memory_pool), embedding=emb)
refinery = Refinery(rfnry_engine.PostgresStorage(engine_pool))
```

A storage of your own is any object with the methods below. `scope` comes first in every one, and
each reads and writes that scope only. `refinery.storage` is the storage the refinery was given, to
read its rows back yourself, and `refinery.record(run)` stores a run and every child run under it —
what an engine with a refinery does after each run not made with `record=False`:

```python
class Storage(Protocol):
    async def check(self) -> None
    async def put_runs(self, scope: str, rows: list[RunRow]) -> None  # every row of one recorded tree, in one transaction
    async def get_run(self, scope: str, run_id: str) -> RunRow | None
    async def runs(self, scope: str, *, run_ids: list[str] | None = None, kind: str | None = None,
                   since: datetime | None = None, unlabeled_by: str | None = None,
                   limit: int | None = 1000) -> list[RunRow]       # newest first
    async def put_label(self, scope: str, row: LabelRow) -> LabelRow  # the row under its id; never replaces one
    async def labels(self, scope: str, *, run_ids: list[str] | None = None,
                     source: str | None = None, since: datetime | None = None) -> list[LabelRow]
    async def run_tree(self, scope: str, *, run_ids: list[str] | None = None,
                       tags: dict[str, str] | None = None) -> list[RunRow]  # oldest first
    async def delete_runs(self, scope: str, *, run_ids: list[str] | None = None,
                          tags: dict[str, str] | None = None) -> int        # the rows it removed
    async def delete_scope(self, scope: str) -> int                 # the rows it removed
    async def count_scope(self, scope: str) -> int

class Refinery:
    storage: Storage
    async def record(self, run: Run) -> None

class RunRow(BaseModel):   scope, run_id, parent_run_id, agent, kind, at, recorded, duration_ms, model, mode, body
class LabelRow(BaseModel): scope, label_id, run_id, source, at, body
class Exported(BaseModel): keys, rows, watermark
class Labeled(BaseModel):  judged, taught, errors
class Conflict(BaseModel): run_id, corrections
```

A row's `body` is the run's record, or the label, as JSON, with a run's children and routed run
left out: each is a row of its own, joined by `parent_run_id`. A generation row's body keeps its
`conversation` and `sent`, the number of its first messages sent at the start, in place of a second
copy as `messages`; `refinery.run` reads `messages` back as `conversation[:sent]`, and
`refinery.record` refuses a run whose `messages` are not the start of its `conversation`. `runs`
returns the newest first and `labels` the oldest first; `unlabeled_by="judge"` keeps the runs no
label from that source names. `run_tree` returns the runs a selector matches — `run_ids`, or `tags`
by containment — and every run under them, oldest first; `refinery.run` reads a run back from it.
`delete_runs` removes the same runs and every label on them in one transaction, for
`refinery.delete_runs`, which checks the selectors before it calls. `Exported.rows` counts the rows
written per dataset; `Conflict.corrections` lists each correction as `(label_id, at, text)`, oldest
first.

## Tenancy

`scope` is the tenant. It is required on every run and every refinery verb, it is the first column
of every table, and no read accepts its absence. Tools pinned to a tenant are passed on the call.
`refinery.delete_scope(scope)` removes everything; `refinery.verify_absent(scope)` proves it.
The tree of agents is shared by every tenant and holds no tenant's data; a run's record holds
copies of what it takes from the tree, and a model adapter copies of those, so no tenant's code
reaches another tenant's runs through them. The workers are shared too: a
script's module state — a client it builds when it loads, a cache it fills, a global it sets — is
shared by every tenant whose runs call it, and only `call` is the tenant's: its `scope` and `tags`
always, and the rest of what it carries — a tool call's `run_id`, `agent`, `tool`, `call_id` and `by`
inside a run, a question's `run_id`, `agent` and `question`. A script that keeps anything between calls keys it by
`call["scope"]`, as the engine keys every row.

Inside a tenant, erase by tag or by run id. A support desk serves every customer in one scope; tag
each run with its customer, and when one asks to be forgotten:

```python
run = await engine.run("support", scope="acme", input=message, tags={"customer": "c-81"})
deleted = await refinery.delete_runs("acme", tags={"customer": "c-81"})
deleted = await refinery.delete_runs("acme", run_ids=[run.run_id])
await refinery.verify_absent("acme", tags={"customer": "c-81"})
```

`delete_runs` takes `run_ids` or `tags`, exactly one, and removes the runs they match, every run
delegated or routed from them, and every label on any of them, in one transaction; it returns the
number of rows removed. A tag matches by containment: `{"customer": "c-81"}` matches a run tagged
`{"customer": "c-81", "arm": "b"}`. `verify_absent` takes the same selectors; without them it checks
the whole scope.

A run a tool started inside another run's call, such as a consult from `engine.tool`, is a run of
its own, and `delete_runs(run_ids=)` does not reach it: ids reach what a run delegated and routed,
not its consults. Erase those by tags. Give `engine.tool(tags=)`, or the `engine.run(tags=)` of your
own tool, the tags of the person it serves, and `delete_runs(tags=)` removes the consult with its
caller. Datasets you exported before are files you hold — drop the customer's rows from
them yourself; every row carries its run's tags.

## At a glance

Everything the package exports, by role.

- **Run things:** `Engine`, `AgentInfo`, `Stop`, `HistoryLike`.
- **What a run returns:** `Generation`, `Decision`, `Run`, `Step`, `Fingerprint`; events
  `RunStarted`, `TextDelta`, `ReasoningDelta`, `MessagesTaken`, `ToolCallStarted`,
  `ToolResultReady`, `CommandOutput`, `Event`.
- **Speak to models:** `GenerationModel`, `StreamingGenerationModel`, `OnDelta`, `DecisionModel`;
  `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`,
  `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`,
  `CheckAnswer`, `Answer`, `Decided`; the shapes a question and a result are read by,
  `QuestionLike`, `ReplyLike`, `DecidedLike`, `AnswerLike`; `confidence`.
- **Give models tools:** `Tool`, `Tools`, `EngineTools`, `current_call`, `CallInfo`, `Jail`,
  `Workspace`, `Backend`, `Process`, `Local`.
- **Refine:** `Refinery`, `Review`, `Outcome`, `Label`, `Labeled`, `Exported`, `Conflict`, `Replay`,
  `Bucket`, `is_holdout`.
- **Store:** `Storage`, `PostgresStorage`, `RunRow`, `LabelRow`.
- **Fail loudly:** `EngineException`, `ConfigurationError`, `NotFoundError`, `InputError`,
  `ModelError`, `RunawayError`, `DelegationLoopError`, `DelegationDepthError`, `StorageError`,
  `WorkerError`.

## Errors

`ConfigurationError` for a tree, a model, a tool or a workspace the engine cannot use — a tree with
no `AGENT.md`, a link to a folder on the way to it or one holding it, one that would reach an agent
twice, a tool's file that links out of its `tools/` folder, a file or folder whose name is not
UTF-8, a file an agent is made of that cannot be read, a question's file or an `AGENT.md` that links
out of the agent's folder, an agent's own `tools/`, `questions/` or `skills/` folder that links out
of it, a `skills:` or `public_skills:` that names a skill the agent does not have, holds a blank,
padded or twice-listed entry or sits in a decision agent's `AGENT.md` (`all` and `none` included), a
`tools:`, `skills:` or `public_skills:` that is an empty list, has no value, is a word but `all`
or `none` or holds an entry that is not text, a `public_skills: all` over an offered skill whose name
holds whitespace, a `public_skills:` entry that holds whitespace, an `AGENT.md` without a `name:` or
a `model:` in plain text, or whose
`kind:` is neither `generation` nor `decision`, a `models` that is not a mapping of names to
adapters, a model no agent runs on, an agent whose model has not its kind's method, two agents with
one name, an agent in `agents:` or `route:` the tree does not have, an unknown key in the
frontmatter of an `AGENT.md` (`frontmatter: unknown key 'route.min_confidance' in
agents/frontdesk/AGENT.md`), frontmatter that is not valid YAML or a `description:` that is not a
string in a file the map would show or a skill, a decision agent with `preload:`, `workspace: true`,
`tools:` or `gitignore:`, a generation agent with `questions/` or `route:`, a `.gitignore` of an
agent with `gitignore: true` that cannot be read or decoded, a `tools:` entry that is blank, padded
or listed twice, a `tools:` that names `Output` or a tool that comes with its own declaration, a
`tools:` name a run does not offer, when the run starts, a tool's `.json` without its `.py` or a
script without its `.json`, a tool's JSON with an unknown key, no description, an `input_schema`
that is not an object or a schema that does not compile, a script or helper that is not valid
Python, imports the engine, the server's own code, a module relatively or a sibling script or
subfolder of its folder, or imports a helper inside a function, a helper named like a
standard-library module, a tool named like one the engine gives the agent, a `workers=` naming a
folder that holds no agent or one inside another it names, a `reads:` that is not a path from the
agent's folder or goes above the root, names no folder of the tree, a `tools/` or `skills/` folder
on an agent's path or a folder inside one, leaves the root, names the agent's own folder, one that
holds it or one inside it, or names two folders that are the same or one inside the other (`tree:
agent 'coding' reads '../shared' and '../shared/', the same folder or one inside the other; keep
one`), `reads:` on a decision agent or on an agent whose folder is a link to a folder elsewhere,
a `preload:` that is empty, has no value or is not a list, an entry of it that is not a path from
the agent's folder, goes above the root, lies outside the agent's folder and the folders it reads or in a folder that holds the agent,
does not exist or cannot be reached, or names the path another entry names or would send a file
another entry sends, a link counted, a file it names that `Read` would refuse or that cannot be
read, one skill as both `skills/x.md` and `skills/x/SKILL.md`, a question in markdown, a question's
JSON that is not an object, holds an unknown key, has no instructions or does not fit its type, a
question's script without its `.json`, a route not by a choice question of the router's
`questions/`, by one a script fills, or that maps an option the question does not have, a route
entry that is not an agent's name (`route.to.technical: input should be a valid string`), or that
gives a decision agent nothing to ask, two tools of a run with one name or one named like a tool the
engine gives the agent, a run of an agent whose model was not given, or of a router whose target's
model was not given, a missing workspace, a `Local` root that is not a directory, a `shell` that is
not a list of strings or whose first word is not on `PATH`, a backend without `spawn`, `read` and
`write`, a `Workspace` without a scope, a `max_steps`, `max_depth`, `tool_timeout`, `history_walk`,
`history_chars`, `timeout` or `output_bytes` that is not positive, a `worker_result_bytes` that is
neither a positive integer nor `None`, an `approve` that returns anything but `True` or `False`, a
judge that is not a decision agent or a teacher that is not a generation agent. `NotFoundError` for
an agent the tree does not have — a judge, a teacher or a replay's agent in the candidate included —
a skill the agent does not have (`engine: agent 'support' has no skill 'refund'; its skills are
escalate`), found when the run starts, or a run the refinery does not. `InputError` for a call the
library refuses: an empty scope, an input that is not a string, a JSON object or a list of JSON
values and parts, a workspace of another scope, a history that breaks a tool call from its result, a
`start` outside the history or without one, tags outside their limits, runtime questions that do not
fit, a runtime id the run already asks, questions for a generation agent, a skill for a decision
agent, a run given a skill both by `skill=` and by an input that starts with a public `/name`
(`engine: a run invokes one skill, by skill= or by an input that starts with /<name>; got
skill='escalate' and /close`), a decision run with no question to ask, media the decision model does
not declare, a media part with both or neither of `data` and `url`, a correction of the wrong shape
or for a question the run was not asked, a `delete_runs` or `verify_absent` given both `run_ids` and
`tags` or an empty one, a `delete_runs` given neither, an `export` or a `replay` given a
`holdout_percent` outside 0 to 100, a `replay` given empty `tags` or a `limit`, `concurrency` or
`repeat` below 1, a tool called with bad arguments, arguments JSON cannot hold or a path outside its
directory, and any use of an engine after `close()` (`engine: this engine was closed; build a new
Engine`). `ModelError` when an adapter breaks the contract — a result whose fields do not fit, a
missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1,
a score outside its levels — or a folder tool's result does not match its `output_schema`, or a
question's script returns anything but an object, a key its JSON sets or one its type does not take,
or a question that does not fit. `RunawayError` when a loop exceeds `max_steps` (default 200).
`DelegationLoopError` for a route into an agent already in the same chain of routes;
`DelegationDepthError` for nesting past `max_depth`, which reaches the model as a tool error through
`Delegate` and raises through a route. `StorageError` for the database. `WorkerError` for a tool or
a question's script that fails in its worker process: what its script raised, `sys.exit()` included,
a result JSON cannot hold, a result or error over `worker_result_bytes`, a worker that stopped while
the call was in flight (`workers/pool: the worker for lookup_order stopped: exited with code 3`) or
a call in flight when `close()` stopped its worker. Inside a run, a tool's error is the model's to
read, never raised. All under `EngineException` — the root takes the `Exception` suffix, every child
ends in `Error`, the same shape as rfnry-memory's. Every message names the module that raised it,
then says what went wrong in lowercase: `engine: agent 'ghost' was not found`, `tools/workspace:
path outside the workspace: ../x`. A tool's error reaches the model with its type in front:
`InputError: tools: unknown tool 'Ghost'`.