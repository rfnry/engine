# rfnry-engine API

The reference: every public name, its signature, what it returns and what it raises. How and why to use them is in [the guide](engine.md).

## Models

### `GenerationModel`

```python
@runtime_checkable
class GenerationModel(Protocol):
    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ReplyLike: ...
```

A model that writes text and calls tools: any object with this method, matched by shape. It returns a `Reply`, or anything with its fields; a result whose fields do not fit makes the run raise `ModelError`. An adapter may also carry `media_types`, the file types its model takes, such as `{"image/*", "application/pdf"}`, so `Load` refuses the rest, and `model`, the name of the model it calls, a `str` the run records as its model when no reply reports one, and `settings`, a JSON object of what it sends its vendor besides the conversation, which the run records beside the model when a reply reports none. Settings are stored and exported, so a credential or an endpoint never goes in them.

### `StreamingGenerationModel`

```python
@runtime_checkable
class StreamingGenerationModel(Protocol):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: OnDelta,
    ) -> ReplyLike: ...
```

A generation model that also streams. When a run has `on_event`, the engine calls `generate_stream` instead of `generate`; the adapter awaits `on_delta(text)` with each piece of the reply's text as it arrives and `on_delta(text, reasoning=True)` with each piece of the model's reasoning its vendor streams, and returns the whole reply, as `generate` does. An adapter that never passes `reasoning=` streams text alone.

### `OnDelta`

```python
class OnDelta(Protocol):
    async def __call__(self, text: str, *, reasoning: bool = False) -> None: ...
```

What `generate_stream` is handed. Text goes out as a `TextDelta`, reasoning as a `ReasoningDelta`, each with the turn. An adapter that annotates `on_delta` as `Callable[[str], Awaitable[None]]` still fits, since it never passes `reasoning`.

### `DecisionModel`

```python
@runtime_checkable
class DecisionModel(Protocol):
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

A model that answers typed questions about a state in one call: any object with this method, matched by shape. It returns an answer per question id, or a `Decided` when it can report what the call cost and which model answered. An adapter may also carry `model`, the name of the model it calls, a `str` the run records as its model when the call reports none, `settings`, a JSON object of what it sends its vendor besides the conversation, recorded when the call reports none (stored and exported, so a credential or an endpoint never goes in them), and `media_types`, the media its model reads in a state, such as `{"image/*"}`: a run's parts with that media reach it as the state, a list of parts; without it the state is text, and parts it does not declare are refused before it is called. Answers that do not fit — a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels — make the run raise `ModelError`.

### `ReplyLike`

```python
class ReplyLike(Protocol):
    text: str
    tool_calls: Sequence[ToolCallLike]
    usage: UsageLike | None
    stop_reason: StopReason
    model: str | None
    settings: Mapping[str, JsonValue] | None
    reasoning: Any
```

The shape a generation model's result is read by: the read-only fields of `Reply`. `ToolCallLike` is a protocol of `id: str`, `name: str` and `args: dict[str, Any]`, `UsageLike` is one of `input`, `output`, `cache_creation` and `cache_read`, each an `int`, and `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `DecidedLike`

```python
class DecidedLike(Protocol):
    answers: Mapping[str, AnswerLike]
    usage: UsageLike | None
    model: str | None
    settings: Mapping[str, JsonValue] | None
```

The shape a decision model's result is read by when it is more than the answers: the read-only fields of `Decided`. `UsageLike` is a protocol of `input`, `output`, `cache_creation` and `cache_read`, each an `int`.

### `AnswerLike`

```python
AnswerLike = ChoiceAnswerLike | ScoreAnswerLike | CheckAnswerLike
```

The shape an answer is read by: the fields of `ChoiceAnswer`, `ScoreAnswer` or `CheckAnswer` without `confidence`, which the engine computes. `ChoiceAnswerLike` is a protocol of `type: Literal["choice"]`, `choice: str` and `probabilities: dict[str, float]`; `ScoreAnswerLike` is one of `type: Literal["score"]`, `score: float` and `probabilities: list[float]`; `CheckAnswerLike` is one of `type: Literal["check"]` and `probability: float`.

### `QuestionLike`

```python
QuestionLike = ChoiceLike | ScoreLike | CheckLike
```

The shape of the questions a decision model is handed: the fields of `Choice`, `Score` or `Check`. `ChoiceLike` is a protocol of `type: Literal["choice"]`, `instructions: str | dict[str, JsonValue] | list[JsonValue]` and `options: Mapping[str, str | dict[str, JsonValue] | list[JsonValue] | None]`; `ScoreLike` is one of `type: Literal["score"]`, `instructions: str | dict[str, JsonValue] | list[JsonValue]` and `levels: Sequence[str | dict[str, JsonValue] | list[JsonValue]]`; `CheckLike` is one of `type: Literal["check"]`, `instructions: str | dict[str, JsonValue] | list[JsonValue]`, `yes: str | dict[str, JsonValue] | list[JsonValue] | None` and `no: str | dict[str, JsonValue] | list[JsonValue] | None`. An adapter branches on `question.type`; one annotated with `QuestionLike` from rfnry-engine or rfnry-memory serves both libraries.

### `Message`

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Content = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False
    prefix: str | None = None
    usage: Usage | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

One turn of a conversation. An assistant turn carries its text in `content`, its calls in `tool_calls` and its `reasoning` as the model returned it; a tool result has `role="tool"`, the `tool_call_id` it answers, and `is_error` when the call failed.

`prefix`, `usage`, `model` and `settings` are the engine's record of an assistant message it made from a reply, never sent to a model and outside the vocabulary shared with rfnry-memory: `prefix` is the first 16 hex characters of a sha256 over everything sent before the message when it was written — the model, as its adapter names it, the system prompt, the tools, the preloaded files and the earlier messages as sent — and its `reasoning` is sent again only while what precedes it is the same; `usage` is what the call that wrote it reported, and `model` the model that wrote it — the one the reply named, else the adapter's `model` attribute — so a run whose adapter changed models mid-run says which wrote each reply, where the run's `model` names the last; `settings` are the JSON object the reply reported, else the adapter's `settings` attribute, and stay out of the `prefix` hash. `prefix` is set on an assistant message whose reply carried reasoning, and is `None` otherwise. A `usage`, `prefix`, `model` or `settings` that does not fit these fields — another library's usage, a prefix that is not 16 hex characters, a model that is not a string, settings that are not a JSON object — reads as `None`, so a message from elsewhere is taken without them; only a `prefix` that is dropped costs its `reasoning`, which is not sent again.

### `HistoryLike`

```python
class HistoryLike(Protocol):
    async def read(self, start: int, stop: int | None = None) -> Sequence[object]: ...
```

A conversation kept anywhere, read by position: the messages at positions `start` up to `stop`, every one from `start` when `stop` is `None`, each read by the fields of `Message`. A position is a message's index in the conversation from 0 and always holds the same message. `Engine.run` and `EngineTools` take one as `history`, as they take a list.

### `ToolSpec`

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
```

A tool as a generation model is offered it: its name, its description and the JSON schema of its input.

### `ToolCall`

```python
class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = {}
```

A call a generation model makes: its id, the tool's name and the arguments.

### `Reply`

```python
class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None
    stop_reason: StopReason = "end_turn"
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
    reasoning: Any = None
```

What a generation model returns. A reply whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty asks the engine to run those calls and call the model again; anything else is final, and a call it holds is answered `NotRun` and never run. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `Usage`

```python
class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0

    def __add__(self, other: Usage) -> Usage: ...
```

The tokens of one model call, or summed over a run; `a + b` adds them field by field. It is frozen, and refuses a negative count or an unknown field.

### `Text`

```python
class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str
```

A text part.

### `Image`

```python
class Image(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: bytes | None = None
    url: str | None = None
```

An image part. It carries exactly one of `data`, whose bytes travel as base64 in JSON, or `url`, which the library never fetches.

Raises `InputError` when given both `data` and `url`, or neither.

### `Audio`

```python
class Audio(BaseModel):
    type: Literal["audio"] = "audio"
    media_type: str
    data: bytes | None = None
    url: str | None = None
```

An audio part, carrying `data` or `url` as `Image` does.

Raises `InputError` when given both `data` and `url`, or neither.

### `Document`

```python
class Document(BaseModel):
    type: Literal["document"] = "document"
    media_type: str
    data: bytes | None = None
    url: str | None = None
    name: str | None = None
```

A document part, such as a PDF, carrying `data` or `url` as `Image` does, and optionally its file name.

Raises `InputError` when given both `data` and `url`, or neither.

### `Part`

```python
Part = Annotated[Text | Image | Audio | Document, Field(discriminator="type")]
```

Any of the four parts, told apart by `type`. A part from another family library is read by its fields.

### `Content`

```python
Content = str | list[Part]
```

What a message or a tool result carries: a string or a list of parts.

### `State`

```python
State = str | dict[str, Any] | list[Any]
```

What a run is given: a string, a JSON object or a list, a list of parts included. It is a generation run's input and what a decision run's questions are about.

### `Choice`

```python
class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]
```

A question answered with one of `options`, each key with what it means, or `None` when the key says enough. It takes at least two options. `instructions` and what an option means are a string, or a JSON object or list for a question with parts or data of its own; `JsonValue` is pydantic's, any value JSON can hold.

### `Score`

```python
class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    levels: list[str | dict[str, JsonValue] | list[JsonValue]]
```

A question answered with a level, `levels` listed in order. It takes at least two levels. `instructions` and each level are a string, or a JSON object or list.

### `Check`

```python
class Check(BaseModel):
    type: Literal["check"] = "check"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None
```

A yes-or-no question; `yes` and `no` may say what each answer means. `instructions`, `yes` and `no` are a string, or a JSON object or list.

### `Question`

```python
Question = Choice | Score | Check
```

Any of the three questions, told apart by `type`.

### `ChoiceAnswer`

```python
class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.0
```

The answer to a `Choice`: the option chosen and a probability per option. The engine sets `confidence` from `probabilities`; an adapter leaves it out.

### `ScoreAnswer`

```python
class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0
```

The answer to a `Score`: a position in `levels`, from 0 to the last index, and a probability per level. The engine sets `confidence` from `probabilities`.

### `CheckAnswer`

```python
class CheckAnswer(BaseModel):
    type: Literal["check"] = "check"
    probability: float
    confidence: float = 0.0
```

The answer to a `Check`: the probability of yes, between 0 and 1. The engine sets `confidence` from it.

### `Answer`

```python
Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer
```

Any of the three answers, told apart by `type`.

### `Decided`

```python
class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

What `decide` returns instead of the answers alone when the adapter can report what the call cost and which model answered. The run's record carries the usage and model the call reported, and its settings; the adapter's `model` and `settings` attributes stand in for what the call does not report.

### `confidence`

```python
def confidence(probabilities: list[float]) -> float: ...
```

Returns how peaked a distribution is, `(n·p_max − 1)/(n − 1)`, at least 0, and 1.0 for fewer than two probabilities. The engine sets every answer's `confidence` with it, so a threshold means the same thing for every adapter and question type.

## Engine

### `Engine`

```python
class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        models: Mapping[str, GenerationModel | DecisionModel],
        workers: Mapping[str, int] | None = None,
        refinery: Recorder | None = None,
        max_steps: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
        history_walk: int = 10_000,
        history_chars: int = 100_000,
    ) -> None: ...

    agents: dict[str, AgentInfo]
    models: dict[str, GenerationModel | DecisionModel]
```

Loads the tree of agents under `root` — every folder that holds an `AGENT.md`, at any depth and `root` itself included, each named by its own `name:`; hidden folders, an agent's `tasks/` and `questions/` and every `tools/` folder are never searched, a folder without an `AGENT.md` is never run, an agent inside another's folder stays out of the outer one's map and preload, and folder links are followed except to a folder of the root, which is walked under its own path — with the tools of its `tools/` folders, each a `<name>.json` and a `<name>.py`, which a generation agent gets from its own `tools/` and from those of the folders above it, the nearest winning a name, and checks it against the models given. `models` maps the names the agents' `model:` give to adapters, `{}` for none; each agent runs on the adapter its `model:` names, and its `preload: true` files are checked against that adapter's `media_types`. `models` is the dict of the mapping given. The tools' scripts, and the scripts beside its decision agents' questions, run in worker processes the engine starts, each when its first call comes, and keeps until `close`; they follow the event loop that calls them, one loop at a time, so a call from another loop stops those the first loop started and starts its own. Workers run on POSIX systems only, Linux and macOS, as `Local` does. `workers` maps folders of the tree, by their paths from the root, to the number of processes of their group: every agent inside a named folder, at any depth, runs its tools in that group, a call going to the process with the fewest calls in flight, and the agents outside the named folders share one process; `None` puts the whole tree in that one. `Recorder` is a protocol of `async def record(self, run: Run) -> None`, which a `Refinery` meets. `tool_timeout` is each tool call's deadline in seconds, `None` for none. `history_walk` is how many messages one `History` search walks back, `history_chars` how many characters one `History` result holds. `agents` maps each agent's name to its `AgentInfo`, in name order. Once loaded, it logs `engine: agents loaded [lead, triage, writer]` at INFO on the `rfnry_engine.engine` logger.

Raises `ConfigurationError` when `root` is not a directory, `models` is not a mapping, a key of it is blank, padded or not text, or a value of it has neither `generate` nor `decide`, a model no agent runs on (`no agent runs on model 'fsat'; the tree's agents run on judge, strong`), an agent whose model lacks its kind's method (`agent 'writer' is a generation agent, and model 'judge' has no generate()`), when it holds no `AGENT.md`, a link leads to a folder that holds the link, or to a folder of the root that holds an `AGENT.md`, in it or below it, or two links reach one such folder outside the root, an `AGENT.md` without a `name:`, a `kind:` or a `model:`, with a `kind:` other than `generation` or `decision`, with a `name:` or `model:` blank, padded or not text, or with a `name:` that is hidden or holds a slash, two agents with one `name:`, a `reads:` that is not a path from the root, when the tree cannot be used (a bad `AGENT.md`, an unknown frontmatter key, frontmatter that is not valid YAML or a `description:` that is not a string in a `.md` file the map would show, a question in markdown (`tree: question agents/triage/questions/urgent.md is markdown; write it as urgent.json with type, instructions and its options, levels or yes and no`), a question's `<id>.json` that is not valid JSON or a JSON object, holds a key but `type`, `instructions`, `options`, `levels`, `yes` and `no`, holds a key of another type (`tree: agents/triage/questions/team.json is a choice question holding 'levels'; a choice takes options, a score levels and a check yes and no, so drop 'levels' or change its type`), has missing or empty instructions or does not fit its type — a choice with fewer than two options, a score with fewer than two levels, unless a script beside it fills them — a question's `<id>.py` without its `<id>.json` (`tree: agents/matcher/questions/order.py has no order.json beside it; a question is order.json, and order.py fills it from data`), a decision agent with `tasks/` (`tree: decision agent 'triage' has tasks/; a decision agent asks only questions/, so make another decision agent for another set of questions`), an agent in `agents:` or `route:` the tree does not have, a route entry naming a task its agent does not have, naming a task for a decision agent or reaching a decision agent with no `questions/`, a route not by a choice question of the router's `questions/`, by a question a script fills (`tree: route of 'triage' is by 'order', whose options come from order.py; route by a question whose options are fixed`), or mapping an option that question does not have, a skill kept as both `skills/<name>.md` and `skills/<name>/SKILL.md`, a decision agent with `tools:`, `tools: []` included, a `tools:` with no list, a `tools:` entry that is blank or padded, a name listed twice in it, a `tools:` that names `Output` (`frontmatter: invalid agent agents/support/AGENT.md; tools: 'Output' cannot be kept by name; it comes with expect= on a run, so leave it out of tools:`), or that keeps a name that is neither a tool the agent sees nor `Read`, `Grep`, `Glob`, `LS` or `Load` (`tree: agent 'support' keeps 'refnd' in tools:, which it does not see; its tools are Glob, Grep, LS, Load, Read, refund`), a tool's `.json` without its `.py` or a `.py` without its `.json`, a tool's JSON that is not an object, holds a key but `description`, `input_schema` and `output_schema`, has no description in text, an `input_schema` whose top level is not `"type": "object"`, or a schema that does not compile, a tool's or a question's script with no `def run` or `async def run` at its top level (`folder: agents/support/tools/lookup_order.py has no run at its top level; a tool defines def run(args, call) or async def run(args, call)`), a tool's or a question's script or helper that is not valid Python or imports `rfnry_engine`, a module whose file lies in the project that holds the tree — the nearest folder at or above `root` with a `pyproject.toml` — outside the standard library, the installed packages and the tree's `tools/` and `questions/` folders, a module relatively, or a helper inside a function or class, a tool named like one the engine gives an agent that sees it (`Read`, `Grep`, `Glob`, `LS`, `Load`, `History`, `Output`, `Delegate` with `agents:`, a workspace's with `workspace: true`), a decision agent with `preload: true` or `workspace: true`, a generation agent with `questions/` or `route:`, a `reads:` naming a folder the tree does not have, a `tools/` folder on an agent's path or a folder inside one, a folder resolved outside the root, the agent's own, one that holds it or one inside it, or two folders one inside the other, `reads:` on a decision agent, `reads:` on an agent whose folder is a link to a folder elsewhere, a file `preload: true` would send that neither `Read` nor `Load` would hand over: over 20 MB, of a type its agent's model's `media_types` does not name, or not the type its name says), when `workers` is not a mapping, names a folder the tree does not have (`engine: workers names 'billing', which is not a folder of the tree; name folders by their path from the root`) or one inside another it names (`engine: workers names 'support/billing' inside 'support', which it names too; keep one`), or gives a folder anything but a whole number of processes of 1 or more, or when `max_steps`, `max_depth`, `tool_timeout`, `history_walk` or `history_chars` is not positive.

### `Engine.run`

```python
async def run(
    self,
    agent: str,
    scope: str,
    input: State,
    *,
    task: str | None = None,
    skill: str | None = None,
    history: Sequence[object] | HistoryLike | None = None,
    start: int = 0,
    context: str | None = None,
    expect: type[BaseModel] | None = None,
    on_event: EventHandler | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    record: bool = True,
    stop: Stop | None = None,
    inbox: asyncio.Queue[Any] | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
) -> Run: ...
```

Runs `agent` for the tenant `scope` on `input` and returns its record: a `Generation` for a generation agent, a `Decision` for a decision agent. A generation agent's `task` puts the body of `tasks/<task>.md`, without its frontmatter, in its system prompt; a decision agent takes none. A generation agent's `skill` puts its skill — `skills/<skill>.md` or `skills/<skill>/SKILL.md` — in front of `input`, in the same user message, as a `Text` of `<<<SKILL <path>>>>` on a line of its own followed by the file's body without its frontmatter; the input follows as its own parts, and an empty input adds none. Delegated and routed runs never take the caller's skill. A decision agent asks the questions of its `questions/`, every `<id>.json` there; one with an `<id>.py` beside it is first filled by that script's `run(input, call)`, in the agent's worker and under `tool_timeout`, the scripts of a run at once — `input` the state as JSON, `call` the run's `scope`, `run_id` and `agent` and the `question` id — whose returned object fills what the JSON leaves out, never its `type` nor another key it sets, and is checked as a question; questions composed at run time, by a model or by a runtime of yours, reach it through `Delegate` or `engine.tool`, read by their fields and asked with the folder's in the same call, and a recorded run's are re-asked by `replay`. `context` is data for this run, such as the contract under review or today's date, never instructions. `expect` is the model the output must fit, its descriptions included, which reach the model in the `Output` tool's schema. `history` is a list of messages or a `HistoryLike`, and the run sends its messages from position `start` on — `start=0` sends them all; a routed run is given both, a delegated run neither. A route's run gets the task its route entry names, if any, never the router's runtime questions. With a refinery attached, a run is recorded unless `record=False` — one that raises after its first model call, or from its `RunStarted` handler, too, as far as it got and with its `error`, before the exception reaches you — and a recording that fails is logged, never raised. `EventHandler` is `Callable[[Event], Awaitable[None]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets, and `tools` rides this run only, a tool of it winning over a folder tool of the same name; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`, called with the agent's name and the call.

Cancelling the task that runs it cancels the model call in flight and every tool call still running, a `Delegate` call and the run it started included, and re-raises the cancellation once all of them have ended, as `asyncio.TaskGroup` does; a tool that ignores cancellation holds it. Each further cancel meanwhile, two that land at the same moment counted as two, does not end the wait: it cancels once more everything still running, so a workspace command still being stopped is killed at once, with no grace period.

Raises `InputError` when the engine was closed (`engine: this engine was closed; build a new Engine`), `scope` is empty, `workspace` belongs to another scope, a decision run has no question to ask, a decision agent is given media its model does not declare in `media_types`, `tags` are outside their limits, `history` does not fit the contract or breaks a tool call from its result (a result whose call is not in it, or a call with no result), `start` is negative, past the end of a list `history`, or given without a `history`, or `input` mixes parts with other values. Raises `InputError` when a decision agent is given a `task` (`engine: decision agent 'grader' takes no task; it asks its questions/, so run another decision agent for another set of questions`) or a `skill`. Raises `NotFoundError` when the tree has no agent `agent`, or the agent has no task `task` or no skill `skill`, before any model is called; the message names the tasks or skills it has. Raises `ConfigurationError`, before any model is called, when the model the agent's `model:` names was not given (`agent 'writer' runs on model 'fast', which this engine was not given`), a decision agent routes to an agent whose model was not given (`agent 'triage' routes to agent 'writer', which runs on model 'fast', which this engine was not given`), an agent with `workspace: true` runs without a `workspace`, `tools` repeats a name of its own or of a tool the engine gives the agent, or holds a tool named `History`, a tool named `Output` collides with `expect`, or `approve` returns anything but `True` or `False`. Raises `ModelError` when an adapter's result does not fit the contract, or a question's script returns anything but an object, returns a key its `<id>.json` sets — `type` always among them, so a script never changes its question's kind (`engine: order.py of decision agent 'matcher' returned 'type', which question 'order' sets in order.json; a script returns only what its question's JSON leaves out, such as {"options": {...}}`) — or a key its type does not take (`engine: order.py of decision agent 'matcher' returned 'levels', which question 'order', a choice, does not take; a choice takes options, a score levels and a check yes and no`), or fills a question that does not fit (`engine: question 'order' of decision agent 'matcher' does not fit once order.py fills it: the question: models: a choice needs at least two options`). Raises `WorkerError` when a question's script raises or its worker stops, naming the script and its question before what it raised (`engine: order.py of decision agent 'matcher' could not fill question 'order': KeyError: 'customer'`), and `TimeoutError` when it passes `tool_timeout` (`workers/pool: order timed out after 30s`); a question's script that fails ends its run before the decision model is called, recorded with its `error`. Raises `RunawayError` when the loop passes `max_steps`. Raises `DelegationLoopError` when a route reaches an agent already in the same chain of routes, and `DelegationDepthError` when a route goes deeper than `max_depth`.

### `Engine.tool`

```python
def tool(
    self,
    agent: str,
    *,
    scope: str,
    name: str | None = None,
    context: str | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
    on_event: EventHandler | None = None,
    stop: Callable[[], Stop | None] | None = None,
) -> Tools: ...
```

Returns a `Tools` of one tool that runs `agent` as a top-level run under `scope`, recorded like any run: named `name` or the agent's name, described by the agent's `description:` in its `AGENT.md`, taking `{"input": str}` — with an optional `task` when the agent has tasks and optional `questions` when it is a decision agent, as `Delegate` takes them — and returning a generation agent's reply or a decision agent's answers as a dict. `context`, `tools`, `workspace`, `tags` and `approve` ride every call. A call made inside a run started with `record=False`, such as a replay's, is not recorded. Called inside a run, each call's run is in that run's foreground: it takes the run's `stop` and `on_event`, as a delegated run does, so the run's `stop` reaches it instead of cancelling the call. `on_event`, given, rides every call instead; `stop`, given, is called at each call and returns the `Stop` of that call's run, or `None`, so a tool that lives as long as a session stops only the turn a person stops. Inside a run, a call whose run a stop ended, or the run it routed to, is answered `stopped by the user`, as a stopped `Delegate` call is; outside one, `dispatch` returns what the stopped run produced. Given inside a run, `on_event` and `stop` win: a `stop` of your own, or `None` from the function, leaves the call out of the run's foreground, and the run's stop cancels it as it cancels any tool. `EventHandler` is `Callable[[Event], Awaitable[None]]`. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, or `tags` are outside their limits. Raises `NotFoundError` when the tree has no agent `agent`, and `ConfigurationError` when its `AGENT.md` has no `description:`. A call raises what `Engine.run` raises, and `InputError` when `questions` does not fit the contract, holds an empty id or an id the run already asks, or is given to a generation agent.

### `Engine.close`

```python
async def close(self) -> None: ...
```

Stops every worker process the engine started, and whatever they started, and returns once each worker has ended, its pipes closed and its process reaped, whether it is called from the event loop its workers serve or from another; a tool call still in flight in one fails with `WorkerError` (`workers/pool: the worker for lookup_order stopped: it was closed`). The engine is closed for good: `run`, a call of an `engine.tool`, `EngineTools`, and a `replay` or `label` over it raise `InputError` (`engine: this engine was closed; build a new Engine`) before any model is called, so nothing is recorded — a replay's pairs and a label's `errors` carry it. A run already in flight goes on, but starts no other run: its later folder tool calls fail as above, a `Delegate` reads as that tool's error (`InputError: engine: this engine was closed; build a new Engine`), and a route fails the decision run with that `InputError`, before the model of the agent it would reach, generation or decision, is called. Leaving `async with Engine(…) as engine:` calls it. Safe to call more than once.

### `Stop`

```python
class Stop:
    def __init__(self) -> None: ...
```

A person's handle on a running run, passed as `run(stop=)`; it reaches the runs it delegates and routes to, and the runs its calls of an `engine.tool` start. A stop set before the run starts ends it before the first model call; one set after it ended does nothing.

### `Stop.now`

```python
def now(self) -> None: ...
```

Cancels the model call in flight and every tool call still running, except a call with a generation run going under the same stop — one a `Delegate` call or an `engine.tool` started, from the call's task or one it made — which is waited for while that run takes the stop, and then cut, with whatever else it was doing, as any call is; a call waiting on a decision run, one model call, is cancelled as a model call in flight is; a call it finds not yet running never runs, so a stop from an `on_event` handler of a call's `ToolCallStarted`, or of a reply's `TextDelta`, comes before the tool runs. A handler still running when a stop cancels its call, or a streamed `TextDelta` or `ReasoningDelta` handler, is cancelled with that work, so a kill switch does its own work, such as an audit, before it calls `stop.now()`. The run returns with `stopped="now"`, and each call it cut off is answered with the error `stopped by the user`. A call that already has its result keeps it, and the stop waits for an `on_event` still handling its `ToolResultReady` to return, so a slow handler delays it; cancelling the run cancels the handler. The run returns once the work the stop cancelled has ended, so a tool that ignores cancellation holds the stop, as it holds a cancel of the run. `now()` is safe to call any number of times: each call after the first, while the run is stopping, cancels once more every tool call still running — not a call with a generation run under the same stop, whose run takes that `now()` itself, nor a call whose `ToolResultReady` handler the stop waits for — so a workspace command still being stopped is killed at once, with no grace period, as it is when the stopping run is cancelled.

### `Stop.after_step`

```python
def after_step(self) -> None: ...
```

Lets the current tool calls finish and ends the run before its next model call, with `stopped="after_step"`. After `now()` it does nothing.

## Records

### `AgentInfo`

```python
class AgentInfo(BaseModel):
    name: str
    path: str
    kind: Kind
    model: str
    description: str
    agents: list[str]
    reads: list[str]
    tools: list[str]
    tasks: dict[str, str]
    skills: dict[str, str]
```

An agent as the engine loaded it, in `Engine.agents`: its name, its `path` — its folder from the root, `.` for the root agent — its kind, its `model` — the name its `AGENT.md` gives — its description, the agents it may call, the folders it reads, by their paths from the root, the `tools` it gets from its folders, those of its own `tools/` and of every folder above it, as its `tools:` keeps them, by name, its tasks and its skills, each to its `description:` — `""` for one without; a decision agent has neither. `Kind` is `Literal["generation", "decision"]`.

### `Generation`

```python
class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    call: CallInfo | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    task: str | None = None
    skill: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    settings: dict[str, JsonValue] | None = None
    fingerprint: Fingerprint
    input: State
    system: str
    preload: list[Part] = []
    start: int = 0
    messages: list[Message]
    conversation: list[Message]
    tools: list[ToolSpec] = []
    steps: list[Step]
    calls: int
    reply: str
    usage: Usage
    children: list[Run] = []
    stopped: Stopped | None = None
    stop_reason: StopReason | None = None
    error: str | None = None
    added: list[Message]
```

The record of a generation run: the `task` and `skill` it was run with, the `context` it was given, when it started (`at`), which agent folder and tools produced it (`fingerprint`), the system prompt as sent, the messages as sent to the first model call — the history, then the input — the whole `conversation`, the tools offered, every tool call as a `Step`, the model calls and their summed usage, the reply, and the runs it delegated to in `children`, in step order. `model` is the model the replies reported; when none did, as in a run stopped during its first model call, it is the adapter's `model` attribute if that is a non-empty string, else `None`. `settings` are those of the call that wrote the last assistant message: the settings its reply reported, else the adapter's `settings` attribute, `None` when neither is a JSON object. An accepted `Output` makes `reply` its JSON. `preload` is what `preload: true` sent, `[]` without it: the files sorted by relative path character by character, a file `Read` returns as a `Text` of `<<<path>>>` on a line of its own followed by what `Read` returns, any other file as a `Text` of `<<<path>>>` followed by the part `Load` returns; every model call receives it as one user message ahead of the messages, and neither `messages` nor `conversation` holds it. `Stopped` is `Literal["now", "after_step"]`; `stopped` is `None` for a run that ended on its own. `start` is the position in the history the run began at, `0` when it was sent whole. `added` is computed from the record: the input and every message after it, `conversation` from the input on — what a caller keeps. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`; `stop_reason` is the last reply's `stop_reason` — `"max_tokens"` for a reply the model's limit cut off, `"tool_use"` after an accepted `Output` — and `None` when no reply came back. `error` is `None` for a run that returned; a run that raised after its first model call, or from its `RunStarted` handler, is recorded with `error` as `"<Type>: <message>"`, `reply` `""` and what it had done by then. `call` is the tool call the run worked for, as its `RunStarted.call` within one scope: the `Delegate` call for a delegated run, the caller's call for an `engine.tool` consult made inside a run, the call a tool body was serving for a run that body or a task it started began, the router's call for a routed run, and `None` outside any tool call or for a call made in another scope.

### `Decision`

```python
class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    call: CallInfo | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    at: datetime
    duration_ms: int
    model: str | None
    settings: dict[str, JsonValue] | None = None
    fingerprint: Fingerprint
    state: State
    questions: dict[str, Question]
    runtime: list[str] = []
    answers: dict[str, Answer]
    usage: Usage
    routed_to: Run | None = None
    error: str | None = None
    added: list[Message]
```

The record of a decision run: the state as the model was handed it — a list of parts when the adapter's `media_types` took them, read back as parts — every question asked, from the folder — a scripted one as its script filled it — and at run time, with `runtime` naming the ids that came at run time, an answer per question id with its `confidence`, the usage, model and settings the adapter reported — its `model` and `settings` attributes when the call reports none — and what produced it (`fingerprint`, whose `tools` is the fingerprint of no tools and whose `questions` covers `questions`). `routed_to` is the run its route started, if any. `added` is the `added` of the run its route started, `[]` when it routed nowhere. `error` is set, and `answers` empty, when the decision model's call or a question's script raised; when a script raised, `questions` holds only the questions that need no script, the runtime ones included; a routed run that failed sits in `routed_to` with its own `error`, or `routed_to` is `None` when the routed run was refused before its first model call, and the router carries the same `error`; a routed run whose `RunStarted` handler raised had started, and sits in `routed_to`. Its `call` is the same as a generation's.

### `Run`

```python
Run = Generation | Decision
```

What `Engine.run` returns, told apart by `kind`.

### `Fingerprint`

```python
class Fingerprint(BaseModel):
    agent: str
    tools: str
    questions: str | None
```

Which agent, which tools and which questions produced a run, each the first 16 hex characters of a sha256. `agent` covers every file in the agent's directory, in the `tools/` folder of each folder above it, under its path from the agent's folder, `../tools/`, and in each folder it names in `reads:` as the engine loaded them, by relative path and content, a read folder's files under its path from the agent's folder, `../<name>/` (`../../shared/books/` for an agent two folders deep), leaving out hidden files and folders (a name starting with `.`), `__pycache__` folders, and links that lead out of their folder; it is computed once, when the `Engine` is built. `tools` covers the tools the model was offered, `Run.tools` sorted by name as canonical JSON: the builtins, `History` when the run's `start` is above 0, the tools the agent gets from its folders, a workspace's, `Delegate` with the agents it may call, the run's `tools=` and `Output` when `expect=` asked for one. `questions` covers the questions a decision run asked, `Decision.questions` as JSON ordered by question id, each question's fields in order and its options and levels as given, runtime questions included; it is `None` for a generation run. None covers the input, the history, `context`, `tags`, the model or the code behind a tool a run passes; a folder tool's script is a file of the agent's.

### `Step`

```python
class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result: Content
    is_error: bool
    at: datetime
    duration_ms: int
    child_run_id: str | None = None
    stopped: bool = False
    approved: bool | None = None
```

One tool call of a generation run: the tool, its arguments, its result, whether it failed, when it finished and how long it took. `child_run_id` names the run a `Delegate` started, `stopped` marks a call `Stop.now()` cut off, or one whose `Delegate` or `engine.tool` run a stop ended, and `approved` is what `approve` answered, `None` when nobody was asked.

### `RunStarted`

```python
class RunStarted(BaseModel):
    run_id: str
    agent: str
    call: CallInfo | None
```

Sent to `on_event` once by every generation run, before anything else it sends — also by a run that a stop set before it started ends before its first model call. `call` is the tool call the run works for, as `current_call` sees it when the run starts: the caller's `Delegate` call for a delegated run, the caller's call for a run an `engine.tool` call started inside a run, the call that started the decision run for a run it routes to, `None` for a run started outside any tool call. It names the call even when it was made in another scope, where the run's record holds `None`. Every later event with this `run_id` belongs under `call`, so sub-agents nest at any depth. A decision run sends no events; the `ToolCallStarted` and `ToolResultReady` of the call that asked it show it. A stop waits for its handler, and one the handler sets ends the run before its first model call; a handler that raises fails the run, recorded with that `error` and no model call.

### `TextDelta`

```python
class TextDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str
```

Text from the model, sent to `on_event` as it is produced; from a model that does not stream, the whole text of each reply. `turn` is the model call of the run that said it, from 0: text of a turn that ended in tool calls was said beside them, and a run that ends in text replies with its last turn's text. A delegated or routed run sends its own `run_id` and counts its own turns.

### `ReasoningDelta`

```python
class ReasoningDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str
```

The model's reasoning, sent to `on_event` as a streaming adapter passes it with `on_delta(text, reasoning=True)`; `turn` as in `TextDelta`. It is not part of the reply or the conversation: a reply's reasoning is recorded as the adapter returned it, in `Message.reasoning`, and reasoning streamed before `Stop.now()` is not kept. A model that does not stream, or does not stream its reasoning, sends none.

### `MessagesTaken`

```python
class MessagesTaken(BaseModel):
    run_id: str
    agent: str
    turn: int
    messages: list[Message]
```

Sent to `on_event` when a run takes messages from its `inbox`, at a step boundary: `messages` are the `role="user"` messages it appended to the conversation, in the order the inbox held them, and `turn` is the model call about to read them. The inbox is first in, first out and the run takes all it holds, so they are the first `len(messages)` items put on it that no earlier event reported. Nothing taken, an inbox not read because `Stop.after_step()` is pending, and messages left when the run ends send none. A stop waits for its handler, and one the handler sets ends the run before that model call, the messages already in its conversation.

### `ToolCallStarted`

```python
class ToolCallStarted(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str
    args: dict[str, Any]
```

Sent to `on_event` when a tool call starts, before any approval, with the arguments the model gave it. A handler that calls `Stop.now()` stops the call before its tool runs, and a handler still running when a stop cancels its call is cancelled with it, so a kill switch does its own work, such as an audit, before it calls `Stop.now()`.

### `ToolResultReady`

```python
class ToolResultReady(BaseModel):
    run_id: str
    agent: str
    call_id: str
    step: Step
```

Sent to `on_event` once, when a tool call is done: `step` is the `Step` the run records for it, equal to its entry in `Generation.steps`, sent on every way a call ends — a result, an error, a timeout, a call `approve` refused, a delegation, a call `Stop.now()` cut off, whether or not it had started, an `Output` included. Any other call the engine answers without running it — an `Output` it accepts or refuses, a call after an accepted `Output`, a call in a reply that ended for a reason other than tool use, answered `NotRun` — sends neither `ToolCallStarted` nor `ToolResultReady`; its step is in `Generation.steps`. A cancelled run raises and is not recorded, so a call it cancels sends no `ToolResultReady`, a call whose tool returns after the cancel included: it ends with the cancellation before any handler sees its result.

### `Event`

```python
Event = RunStarted | TextDelta | ReasoningDelta | MessagesTaken | ToolCallStarted | ToolResultReady
```

What `on_event` receives.

### `Review`

```python
class Review(BaseModel):
    stance: Literal["approval", "objection"] | None = None
    comment: str | None = None
    correction: str | dict[str, Any] | None = None
```

A person's label on a run, for `Refinery.review`: a stance, a comment, a correction, or several of them; it needs at least one. A generation run takes a text correction, a decision run a value per question id.

### `Outcome`

```python
class Outcome(BaseModel):
    status: Literal["succeeded", "failed"]
    detail: str | None = None
```

What the world reported about a run, for `Refinery.outcome`.

### `Label`

```python
class Label(BaseModel):
    label_id: str
    run_id: str
    scope: str
    at: datetime
    source: Literal["human", "world", "judge", "teacher"]
    review: Review | None = None
    outcome: Outcome | None = None
    answers: dict[str, Answer] | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

A stored label: its `source` and what it says — a person's `review`, the world's `outcome`, or a judge's or a teacher's `answers`. A judge's or a teacher's label also names the `model` and `settings` of the run that made it; a person's and the world's leave them `None`.

### `Labeled`

```python
class Labeled(BaseModel):
    judged: int = 0
    taught: int = 0
    errors: dict[str, str] = {}
```

What `Refinery.label` returns: how many runs the judge and the teacher labeled, and each run that could not be labeled, by run id, with why.

### `Exported`

```python
class Exported(BaseModel):
    keys: list[str]
    rows: dict[str, int]
    watermark: datetime | None
```

What `Refinery.export` returns: the keys written, the rows written per dataset, and the `watermark` to pass as the next export's `since`.

### `Conflict`

```python
class Conflict(BaseModel):
    run_id: str
    corrections: list[tuple[str, datetime, str]]
```

A run that carries two different human corrections: each as `(label_id, at, text)`, oldest first.

### `Replay`

```python
class Replay(BaseModel):
    recorded: Run
    labels: list[Label]
    attempt: int = 0
    candidate: Run | None = None
    error: str | None = None
```

One pair from `Refinery.replay`: the recorded run as `Refinery.run` reads it back, every label on it oldest first, which of the `repeat` replays of that run this is, from 0, and the candidate's run on what the recorded run was given, never recorded. When the candidate failed, `candidate` is `None` and `error` is the failure as `"TypeName: message"`.

## Tools

### `Tool`

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    body: Body | None = None
    args: type[BaseModel] | None = None
    input_schema: dict[str, Any] | None = None
```

One tool: a name, a description, an input schema and a body. `Body` is `Callable[[dict[str, Any]], Awaitable[Result]]`, an async function of the arguments, where `Result` is `str | dict[str, Any] | list[Part]`; `args` validates the arguments and gives the schema, its descriptions and docstring included, unless `input_schema` does; a tool given only `input_schema` receives its arguments as the model sent them. `description` is what the model reads about the tool, and its first line is what the prompt's list of tools shows. It runs in the consumer's process and rides a run, in `run(tools=)`, where it wins over a folder tool of the same name, and nothing checks it for a body or a description before it is called.

### `Tool.schema`

```python
def schema(self) -> dict[str, Any]: ...
```

Returns the input schema: `input_schema`, else the JSON schema of `args`, else an object with no properties.

### `Tool.definition`

```python
def definition(self) -> dict[str, Any]: ...
```

Returns the tool as a model is offered it, `{name, description, input_schema}`.

### `Tool.call`

```python
async def call(self, args: dict[str, Any]) -> Result: ...
```

Validates `args` against `args` when the tool has one, awaits the body, and returns what it returns. `Result` is `str | dict[str, Any] | list[Part]`.

Raises `ConfigurationError` when the tool has no body. Raises `InputError` when the arguments do not fit `args`.

### `Tools`

```python
class Tools:
    def __init__(self, tools: Iterable[Tool | ToolsLike] = ()) -> None: ...

    names: list[str]

    def __add__(self, other: Tools | ToolsLike) -> Tools: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
```

A set of tools with unique names: what a model receives, and what code can call the same way. It takes `Tool`s and anything else with its two verbs: `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which another `Tools` meets. `names` lists the tools in order and `a + b` is the union.

Raises `ConfigurationError` when two tools share a name, on `+` as well.

### `Tools.definitions`

```python
def definitions(self) -> list[dict[str, Any]]: ...
```

Returns each tool as `{name, description, input_schema}`, in order.

### `Tools.dispatch`

```python
async def dispatch(self, name: str, args: dict[str, Any]) -> Result: ...
```

Calls the tool `name` with `args` and returns its result. `Result` is `str | dict[str, Any] | list[Part]`.

Raises `InputError` when there is no tool `name`. A tool's own errors pass through.

### `Tools.only`

```python
def only(self, *names: str) -> Tools: ...
```

Returns a `Tools` of the named tools, in the order named.

Raises `InputError` when a name is not in the set.

### `Tools.without`

```python
def without(self, *names: str) -> Tools: ...
```

Returns a `Tools` without the named tools; a name not in the set is ignored.

### `EngineTools`

```python
class EngineTools(Tools):
    def __init__(
        self,
        engine: Engine,
        *,
        agent: str,
        scope: str,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
        history: Sequence[object] | HistoryLike | None = None,
        start: int = 0,
        on_event: EventHandler | None = None,
        stop: Callable[[], Stop | None] | None = None,
    ) -> None: ...
```

The tools `agent` runs with under `scope`, as a `Tools` to drive without a model, exactly as a run with these arguments offers them: the builtins `Read`, `Grep`, `Glob`, `LS` and `Load`, `History` when `start` is above 0, the tools it gets from its folders — but one `tools` holds a tool of the same name, which wins — the workspace's tools when it declares `workspace: true`, `Delegate` when it names `agents:`, and `tools`; then `tools:` in its `AGENT.md` keeps only the folder tools and builtins it names — none for `tools: []` — while `History`, the workspace's tools, `Delegate` and `tools` stay, each coming with its own declaration. A folder tool's call has its arguments checked against its `input_schema` — a mismatch an `InputError`, `declared: invalid arguments for lookup_order: 5 is not of type "string" at order_id` — and goes to the agent's worker with `call` holding `scope` and, inside a run's tool call of this scope, the call's `run_id`, `agent`, `tool` and `call_id`, under the engine's `tool_timeout`, which counts the wait for a worker too (`TimeoutError: workers/pool: lookup_order timed out after 30s`); arguments JSON cannot hold, a `datetime` or a `set` passed from code, are an `InputError` before any worker sees them (`workers/pool: the arguments for lookup_order are not JSON: Object of type datetime is not JSON serializable`); its result is checked against its `output_schema`, a mismatch a `ModelError`, and a list whose every item has the `type` of a part comes back as parts, its media `data` read from base64. What the script raised, a result JSON cannot hold, and a worker that stopped are a `WorkerError`. The workers follow the event loop that calls them, one loop at a time: a call from another loop stops the workers the first one started and starts its own. `Delegate` takes `agent` and `input`, an optional `task` when one of the agents it lists has tasks, and optional `questions` — ids to objects with the fields of `Choice`, `Score` or `Check` — when one is a decision agent; its description lists each agent's name, description, kind and tasks. A `Delegate` dispatched through it runs the child as a top-level run — not recorded when it is dispatched inside a run started with `record=False` — with this scope, `tools`, `workspace`, `tags` and `approve`, and the `task` and `questions` it is called with, its `on_event` and `stop` as `engine.tool` takes them (`EventHandler` is `Callable[[Event], Awaitable[None]]`), and raises what `Engine.run` raises, and `InputError` when `questions` does not fit the contract, holds an empty id or an id the run already asks, or is given to a generation agent; inside a run the same refusals reach the model as tool errors. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. `History` reads `history` before position `start`: a page of up to `limit` messages — 20, at most 100 — before `before`, which defaults to `start` and never passes it, each under a JSON header with its `position`, `role`, and its `tool_calls`, or `tool_call_id` and `is_error`, in the order they happened; with `pattern`, a regular expression (`case_insensitive`), only the messages whose text matches, walking back at most the engine's `history_walk` messages. An image, audio or document shows as a line naming it — its type, its media type, a document's `name` and its `url` when it has one — never as the file. A result holds at most `history_chars` characters of messages and, when there is more, ends with where to go on; the engine's `tool_timeout` bounds a search's pattern.

Raises `InputError` when the engine was closed (`engine: this engine was closed; build a new Engine`), `scope` is empty, `workspace` belongs to another scope, `start` is negative, `start` is above 0 without a `history`, or `start` is past the end of a list `history`. Raises `NotFoundError` when the tree has no agent `agent`. Raises `ConfigurationError` when the agent declares `workspace: true` and no `workspace` is given, two tools of `tools` share a name, or one shares a name with a tool the engine gives the agent. A tool named `History` in `tools` is a `ConfigurationError` whatever `start` is: the name is the builtin's on every run.

### `current_call`

```python
def current_call() -> CallInfo | None: ...
```

The tool call the code runs inside: a tool's body, the run's `approve` and `on_event` handlers for that call, and the tasks the body starts — they keep answering for it after it ends. `None` outside a run's tool call: a `Tools.dispatch` or an `EngineTools` call of your own. A delegated run's calls, and the calls of a run an `engine.tool` call starts, name their own run; the rest of such a run — its model calls and its text handlers — runs inside the caller's call and sees that one.

### `CallInfo`

```python
class CallInfo(BaseModel):
    run_id: str
    agent: str
    scope: str
    tool: str
    call_id: str
```

What `current_call` returns, and what `RunStarted.call` holds: the run's id, its agent and scope, the tool called and the id of the call, as `ToolCall.id` carries it. Frozen, so it can be kept and compared.

### `Jail`

```python
class Jail:
    def __init__(self, root: Path | str, *, also: Iterable[Path | str] = ()) -> None: ...

    root: Path
    also: list[Path]
```

Keeps paths inside one directory, `root`, or inside one of `also` — the folders an agent reads — all resolved; for a tool of your own that reads the agent's directory, as the builtins do. It notes each place's device and inode when it is made and checks a place by them before it opens anything in it. The check covers `root` and every folder of `also`, so it guards an agent's folders as `Local` guards its root; it compares directory inode numbers, so a filesystem where they are not stable — overlayfs without `xino`, FUSE without `use_ino` — is not supported for any place, and a `Jail` made before its place exists refuses every open in it.

### `Jail.resolve`

```python
def resolve(self, path: str | Path) -> Path: ...
```

Returns `path`, taken relative to `root` unless it is absolute, resolved with its symlinks.

Raises `InputError` when it resolves outside `root` and every place in `also`.

### `Jail.open`

```python
def open(self, path: str | Path, flags: int = 0, mode: int = 0o666) -> int: ...
```

Opens `path`, resolved as `resolve` does, and returns its file descriptor, which the caller closes. `flags` and `mode` are those of `os.open`; `flags` is `os.O_RDONLY` unless given. It opens the place that holds the target, checks it is still the folder the `Jail` was made with, and walks down from it one folder at a time without following a link, so a link swapped in after the check is refused; with `os.O_CREAT` it creates the missing folders on the way. The last name is opened non-blocking, so a pipe never holds the call. The builtins and `Local` read and write through it.

Raises `InputError` when `path` resolves outside `root` and every place in `also`, when the walk meets a link, or when the place is no longer the folder it was. What the system refuses, such as a missing file, is raised as the `OSError` it is.

### `Workspace`

```python
class Workspace:
    def __init__(
        self,
        backend: Backend,
        *,
        scope: str,
        timeout: float = 120.0,
        output_bytes: int = 30_000,
    ) -> None: ...

    backend: Backend
    scope: str
    timeout: float
    output_bytes: int
```

Where an agent with `workspace: true` writes code and runs it, passed as `run(workspace=)`: the library's tools over `backend`, for the tenant `scope`. `timeout` bounds every `Shell`, `Job`, `Write` and `Edit` call in seconds; a command still running at the limit, or when its call is cancelled, is stopped with its process's `kill()`, and so is anything a command leaves running when it ends, after which what remains of its output is read for at most a second. On `Local`, whose `read` and `write` each run in a thread a cancel cannot stop, a `Write` or `Edit` past `timeout`, like a workspace `Read` or `Load` past `Engine(tool_timeout=)`, returns its `TimeoutError` only once the thread has ended. Every kill runs to its end and the call that started it waits for it, however many cancels come meanwhile, so a stop or a cancel of `Shell` or `Job(id, stop=true)` returns once the kill has returned. A kill counts as the first stop, and two cancels that land at the same moment count as two: the next cancel after the one that started it, or the second during a kill the workspace started on its own — for a command past `timeout` or that has ended, `Job(id, stop=true)` or `close` — cancels that kill, once, so `kill()` ends the command at once. A `kill()` error the call cannot raise, because it is already leaving with a cancellation or with the error of a `wait()` that raised, is logged as an error, `workspace: could not stop the command: …`; one from the kill that follows a job's end is in the job's output. `output_bytes` caps `stdout` and `stderr` each, keeping their first and last half. Jobs belong to the workspace and outlive the runs that started them; each is read in the background until it ends or `close` stops it. A workspace and its jobs live in one event loop: a job whose loop ends is stopped.

Raises `ConfigurationError` when `backend` lacks `spawn`, `read` or `write`, `scope` is empty, or `timeout` or `output_bytes` is not positive.

### `Workspace.tools`

```python
def tools(self) -> list[Tool]: ...
```

Returns the five tools that an agent with `workspace: true` gets whenever a workspace is passed: `Shell(command, background=false)`, which returns `exit_code`, `stdout`, `stderr` and `timed_out`, or with `background=true` starts a job and returns `{"job": id}` at once; `Job(id, wait=0, stop=false)`, which waits up to `wait` seconds for the job to end, then returns its `output` since the last read, `running`, and `exit_code` once it has ended, and with `stop=true` stops a job still running after the wait; `Jobs()`, which lists the jobs as `{"jobs": [{job, command, started, running, exit_code}]}`; `Write(path, content)`, which writes a file exactly; and `Edit(path, old, new, all=false)`, which replaces one exact occurrence, or every occurrence with `all=true`, in a UTF-8 file of at most 5 MB, and says how many it replaced. They go through the backend, and its errors reach the caller as they are. `Job` raises `InputError` for a `wait` outside 0 to `timeout`, or an id with no job: a job leaves the list once its last output is read.

### `Workspace.close`

```python
async def close(self) -> None: ...
```

Stops every job of the workspace with its process's `kill()`, waits for each to end, and empties the list; a job whose `kill()` raised stays listed, so a second `close` or `Job(id, stop=true)` can reach it. The workspace stays usable and its ids keep counting. Leaving `async with Workspace(…) as workspace:` calls it. Raises the first error a `kill()` raised, after asking every job, and logs the others. A cancelled `close` returns once every `kill()` has returned — a second cancel cancels each `kill()` still running, once — and logs the `kill()` errors it cannot raise; cancelled while its kills run, it still empties the list of the jobs they stopped and waits for those to end, as an uncancelled `close` does, before it raises. Each job still ending takes the cancels of `close` as its own — a further one that came before the wait, as the wait starts — so a first lets the kill that follows the job's end run its course, and a further one cancels that kill, once.

### `Backend`

```python
@runtime_checkable
class Backend(Protocol):
    async def spawn(self, command: str) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...
```

Where a workspace's commands run and its files live, passed to `Workspace`. `spawn` starts `command` in the workspace's directory with no input and returns at once, and must return promptly: a stop or a cancel that comes while it runs waits for it to return, then for the kill of what it started, however many cancels come meanwhile, and a second one cancels that kill as soon as it starts; with `background=true` it waits for the job to be listed, so `close` stops it. `read` returns a file's bytes and refuses one larger than `limit`; `write` writes a file whole, creating its parent directories. Paths are the model's, relative to the workspace, and what they can reach is the backend's to decide. An error a backend raises reaches the model as the tool's error. A backend's description, if it has one, is ignored: what the environment holds is told in the agent's markdown.

### `Process`

```python
class Process(Protocol):
    stdout: Stream
    stderr: Stream

    async def wait(self) -> int: ...

    async def kill(self) -> None: ...
```

A command a backend started. `stdout` and `stderr` are read until they return `b""`; `Stream` is a protocol of `async def read(self, n: int) -> bytes`, which an `asyncio.StreamReader` meets. `wait` returns the exit code once the command has exited, even while something it started still holds its output open. `kill` ends the command and everything it started and returns once they have ended or cannot be stopped further; it may be called after the command exited, to end what it left behind, and more than once. The workspace waits for every `kill` it starts, through any cancel, so it must return promptly, and it does not retry one that raised: a `kill` owns its escalation, as `Local` sends SIGKILL a second after SIGTERM. A second stop — a cancel that comes while the workspace waits for a `kill`, after the one that started it, or the second during a kill the workspace started on its own — cancels it, once, and the workspace still waits for it: a cancelled `kill` must still end the command and everything it started, at once, before it returns, as `Local` sends SIGKILL in a `finally`. Its error is the error of the call that asked for it, is logged when that call has another error to raise or is leaving with a cancellation, and is in the job's output when it follows a job's end.

### `Local`

```python
class Local:
    def __init__(
        self,
        root: Path | str,
        *,
        shell: Sequence[str] = ("bash", "-c"),
        env: dict[str, str] | None = None,
    ) -> None: ...

    root: Path
    shell: list[str]
    env: dict[str, str]

    async def spawn(self, command: str) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...
```

A `Backend` on this machine. Commands run as `[*shell, command]` in `root`, in a new session, with `env` as their whole environment — by default `PATH`, `HOME` set to `root`, and `LANG`; a sandbox goes in front of the shell, as in `shell=["bwrap", …, "bash", "-c"]`. `kill` sends the command's process group SIGTERM and, a second later or at once when it is cancelled, SIGKILL. `read` and `write` stay inside `root`, symlinks resolved, and each runs in a thread: a call cancelled meanwhile returns once its thread has ended, however many cancels come, so a `Write` or `Edit` cancelled while it writes has written its file when the cancel returns.

Raises `ConfigurationError` when `root` is not a directory, `shell` is not a non-empty list of non-empty strings, or its first word is not on the `PATH` of `env`. `spawn` raises `InputError` when the command cannot start, as when `root` was removed: `workspace: cannot start the command: No such file or directory`. `read` and `write` raise `InputError` for a path outside `root`, through a link that leads out of it or one swapped in while they walk to the file, and for every path once `root` is no longer the folder it was when `Local` was made — replaced, or reached through a link swapped in for it or a folder above it; `read` for a path that is not a file — missing, a directory, a FIFO or a socket, or a path through a file, such as `notes.txt/x` — or a file larger than `limit`; `write` for a directory, a FIFO or a socket, and a path through a file as `workspace: cannot write notes.txt/x: Not a directory`. Any other error the system gives is `workspace: cannot read {path}: {reason}` or `workspace: cannot write {path}: {reason}`. Every message names the path as the model gave it, never the path on this machine. The root check compares directory inode numbers, so a filesystem where they are not stable — overlayfs without `xino`, FUSE without `use_ino` — refuses a valid root and is not supported as a workspace root.

## Storage

### `Storage`

```python
@runtime_checkable
class Storage(Protocol):
    async def initialize(self) -> None: ...

    async def put_runs(self, scope: str, rows: list[RunRow]) -> None: ...

    async def get_run(self, scope: str, run_id: str) -> RunRow | None: ...

    async def runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        kind: str | None = None,
        since: datetime | None = None,
        unlabeled_by: str | None = None,
        limit: int | None = 1000,
    ) -> list[RunRow]: ...

    async def put_label(self, scope: str, row: LabelRow) -> LabelRow: ...

    async def labels(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> list[LabelRow]: ...

    async def run_tree(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[RunRow]: ...

    async def delete_runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> int: ...

    async def delete_scope(self, scope: str) -> int: ...

    async def count_scope(self, scope: str) -> int: ...
```

The one storage protocol, as the refinery calls it; `PostgresStorage` implements exactly these methods, as a storage of your own does. Every method but `initialize` takes `scope` first and reads and writes that scope only: `initialize` creates what the storage needs and is safe to run again; `PostgresStorage` refuses a table an earlier version made, naming the columns it lacks; `put_runs` stores every row of one recorded tree in one transaction, replacing rows with the same ids; a row of another scope raises `StorageError` and nothing is stored; `put_label` stores a row unless the scope already holds a label with its id, which it never replaces, and returns the row stored under that id, so calls racing on one id all get the same row; `get_run` returns a run or `None`; `runs` returns runs newest first by `at`, at most `limit` or all with `None`, narrowed by `run_ids`, `kind`, `since` (stored after it, by `recorded`) and `unlabeled_by`, which keeps the runs no label from that source names; `labels` returns labels oldest first, narrowed by `run_ids`, `source` and `since`; `run_tree` returns the runs `run_ids` or `tags` (by containment) match and every run under them, oldest first; `delete_runs` removes those runs and every label on them, and `delete_scope` every run and label in the scope, each returning the number of rows removed; `count_scope` counts the scope's runs and labels. A storage raises `StorageError` when a row's scope is not the call's, and `PostgresStorage` raises it whenever the database refuses.

### `PostgresStorage`

```python
class PostgresStorage:
    def __init__(self, pool: Any) -> None: ...
```

Implements `Storage` in Postgres, over an asyncpg pool you built. `initialize` creates the tables `engine_runs` and `engine_labels` and their indexes, so one database can hold other libraries' tables beside them; `delete_runs` and `delete_scope` each run in one transaction; `put_label` inserts unless the id is taken and then reads the stored row back, so processes racing on one id store one row and all get it.

### `RunRow`

```python
class RunRow(BaseModel):
    scope: str
    run_id: str
    parent_run_id: str | None
    agent: str
    kind: str
    task: str | None
    at: datetime
    recorded: datetime
    duration_ms: int
    model: str | None
    body: dict[str, Any]
```

A stored run. `body` is the run's record as JSON with its children and routed run left out, each a row of its own joined by `parent_run_id`, and a generation's `messages` replaced by `sent`, their count, since they are the start of its `conversation`; read a run back with `Refinery.run`; `task` is a generation's task, `None` for a decision, which takes none; `at` is when the run started and `recorded` when the refinery stored it.

### `LabelRow`

```python
class LabelRow(BaseModel):
    scope: str
    label_id: str
    run_id: str
    source: str
    at: datetime
    body: dict[str, Any]
```

A stored label; `body` is the `Label` as JSON.

### `Refinery`

```python
class Refinery:
    def __init__(self, storage: Storage) -> None: ...

    storage: Storage
```

Records runs, stores labels on them, exports datasets and replays recorded runs on a candidate, all through `storage`, which stays readable to read the rows back yourself. Passed as `Engine(refinery=)`, it records every run that returns. A storage's `StorageError` passes through every verb.

### `Refinery.initialize`

```python
async def initialize(self) -> None: ...
```

Initializes the storage: for `PostgresStorage`, creates its tables. Safe to run again.

### `Refinery.record`

```python
async def record(self, run: Run) -> None: ...
```

Stores `run` and every run delegated or routed under it, all stamped with one `recorded` time: what an engine with a refinery does after each run not made with `record=False`.

Raises `InputError` when the run's scope is empty, or a generation's `messages` are not the start of its `conversation`; nothing is stored then.

### `Refinery.run`

```python
async def run(self, scope: str, run_id: str) -> Run: ...
```

Returns the recorded run as `Engine.run` returned it, its delegated and routed runs back in place; a child's id returns that child and the runs under it.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.review`

```python
async def review(
    self,
    scope: str,
    run_id: str,
    review: Review,
    *,
    label_id: str | None = None,
) -> Label: ...
```

Stores a person's review of the run and returns the label, with `source="human"`. Without `label_id` the label gets a fresh id. With it the label is stored under `label_id`, once: when the scope already holds a label with that id on this run, that stored label is returned and nothing is written, so the same call made again — after a crash, or by every process that heard the same event, at the same moment or not — stores one label.

Raises `InputError` when `scope` or `label_id` is empty, when the correction does not fit the run — a dict for a generation run, a string for a decision run, or a question id the run was not asked — or when the scope's label `label_id` is on another run. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.outcome`

```python
async def outcome(
    self,
    scope: str,
    run_id: str,
    outcome: Outcome,
    *,
    label_id: str | None = None,
) -> Label: ...
```

Stores what the world reported about the run and returns the label, with `source="world"`. `label_id` works as in `Refinery.review`: the label is stored under it once, and a later call with it on the same run returns the stored label and writes nothing.

Raises `InputError` when `scope` or `label_id` is empty, or when the scope's label `label_id` is on another run. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.conflicts`

```python
async def conflicts(self, scope: str) -> list[Conflict]: ...
```

Returns the runs that carry two different human corrections. A dataset can hold either, not both; the library reports rather than guesses.

Raises `InputError` when `scope` is empty.

### `Refinery.export`

```python
async def export(
    self,
    scope: str,
    bucket: Bucket,
    *,
    since: datetime | None = None,
    holdout_percent: int = 20,
) -> Exported: ...
```

Writes every run recorded or labeled after `since`, with all its labels, to `bucket` as JSONL under `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, in the datasets `generation`, `preference` and `decision`. Every row carries, after `run_id`, its run's `parent_run_id` and its `call` as `Run.call` holds it, `holdout`, from a hash of its run id and `holdout_percent`, and its run's `fingerprint` as `{"agent", "tools", "questions"}`. A generation row carries the run's `stopped` and `stop_reason`. A run whose `error` is set is in no dataset, except a decision run that answered before the run it routed to raised: its rows are written. Returns the keys, the rows per dataset and the watermark for the next `since`.

Raises `InputError` when `scope` is empty or `holdout_percent` is outside 0 to 100, before anything is read or written.

### `Refinery.label`

```python
async def label(
    self,
    scope: str,
    engine: Engine,
    *,
    judge: str | None = None,
    teacher: str | None = None,
    limit: int = 500,
) -> Labeled: ...
```

Runs `judge`, a decision agent, over the generation runs it has not labeled, and `teacher`, a generation agent, over the decision runs it has not labeled — at most `limit` runs each, their own runs skipped — and stores each answer as a label. A run that failed is labeled like any other; the judge is told its `error`. A run that could not be labeled is named in `errors` and stays unlabeled for the next call.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the tree has no agent `judge` or `teacher`. Raises `ConfigurationError` when `judge` is not a decision agent or `teacher` not a generation agent; each check happens before any run is labeled.

### `Refinery.replay`

```python
def replay(
    self,
    scope: str,
    candidate: Engine,
    *,
    agent: str,
    holdout: bool = False,
    labeled: bool = False,
    tags: dict[str, str] | None = None,
    fingerprint: Fingerprint | None = None,
    limit: int | None = None,
    holdout_percent: int = 20,
    expect: type[BaseModel] | None = None,
    approve: Approve | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    concurrency: int = 1,
    repeat: int = 1,
    conversation: Callable[[Run], Sequence[object] | HistoryLike] | None = None,
) -> AsyncGenerator[Replay]: ...
```

Runs `candidate` on what the top-level runs of `agent` in `scope` were given and yields a `Replay` for each, oldest first by when the run started. `holdout` keeps the held-out runs, by the same hash and `holdout_percent` as `export`; `labeled` the runs with a `human` or `world` label; `tags` the runs whose tags contain these, as `delete_runs` matches; `fingerprint` the runs whose fingerprint equals it, all three parts; `limit` the first so many. Each replay is `candidate.run(agent, scope, …, record=False)` with the recorded input — parts read back as parts — the history (the messages sent before the input) with `start=0`, so without `History` — or, given `conversation`, what it returns for the recorded run, cut before the recorded input, with the recorded `start`, so the candidate has the same window and `History` over everything before it — the task, the skill, which the candidate reads from its own folder, and the `context`; a decision run replays its state and its runtime questions as recorded, the rest of its questions coming from the candidate's folder, its scripts filling them anew, and when its route reached a generation run, that run's input, history and context. `expect`, `approve`, `tools` and `workspace` are passed to every replay. `conversation` is called once per replay, with the recorded top-level run, and applies to the generation run it replays; a decision run that reaches no generation run never calls it, and a `conversation` that raises, or that returns a conversation ending before the recorded input, gives the pair its `error`. It gives the recorded window only when the run was sent its conversation with `start`: a run recorded with `start=0` on a list the app had already cut replays on the conversation's first messages instead. `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. The candidate's tools run for real. Each run is replayed `repeat` times, each attempt a run of its own yielded as a pair of its own whose `attempt` counts from 0; the attempts of a run come one after another, in attempt order, and `limit` counts recorded runs, not attempts. At most `concurrency` replays run at once, attempts included, and while a pair is being read at most `concurrency - 1` run, so with the default a replay starts when its pair is asked for. Pairs are yielded in that order either way. Closing the generator, or cancelling its reader while it waits for a pair, cancels each replay still running and waits for them to end. That close stands for the reader's first cancel: each cancel of the reader beyond its first, two at the same moment counted as two, cancels them once more, so a second ends a replay's workspace command at once. The close returns once they have ended, and a cancelled reader then raises `CancelledError`. A candidate that raises yields its pair with `error` and the replay goes on, the other attempts of the same run included. Nothing is recorded and no label is written.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, `holdout_percent` is outside 0 to 100, `tags` is empty, or `limit`, `concurrency` or `repeat` is below 1. Raises `NotFoundError` when the candidate has no agent `agent`. Each check happens when `replay` is called, before anything is read; a storage's `StorageError` is raised by the first pair.

### `Refinery.delete_scope`

```python
async def delete_scope(self, scope: str) -> int: ...
```

Removes every run and label in the scope, and returns the number of rows removed.

Raises `InputError` when `scope` is empty.

### `Refinery.delete_runs`

```python
async def delete_runs(
    self,
    scope: str,
    *,
    run_ids: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> int: ...
```

Removes the runs `run_ids` or `tags` match, a tag by containment, every run delegated or routed from them, and every label on any of them, and returns the number of rows removed.

Raises `InputError` when `scope` is empty, when given neither selector or both, or when the one given is empty.

### `Refinery.verify_absent`

```python
async def verify_absent(
    self,
    scope: str,
    *,
    run_ids: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> bool: ...
```

Returns `True` when nothing the selector matches is left: no run, no run under one, and for `run_ids` no label. Without a selector it checks that the scope holds nothing.

Raises `InputError` when `scope` is empty, when given both selectors, or when the one given is empty.

### `Bucket`

```python
@runtime_checkable
class Bucket(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...
```

Where `Refinery.export` writes: any object with this method. Keys never repeat, so an export never overwrites an earlier one.

### `is_holdout`

```python
def is_holdout(run_id: str, *, holdout_percent: int = 20) -> bool: ...
```

Returns whether the run is held out, from a hash of its id: the same answer every export gives its rows.

Raises `InputError` when `holdout_percent` is outside 0 to 100.

## Errors

### `EngineException`

```python
class EngineException(Exception): ...
```

The root of every error the library raises. Every message names the module that raised it, then says what went wrong in lowercase: `engine: agent 'ghost' was not found`. Inside a run, a tool's error is the model's to read, with its type in front, and is never raised.

### `ConfigurationError`

```python
class ConfigurationError(EngineException): ...
```

A tree, a model, a tool or a workspace the engine cannot use, found when it is built or when a run needs it.

### `NotFoundError`

```python
class NotFoundError(EngineException): ...
```

An agent the tree does not have, a judge or a teacher included, a task the agent does not have, or a run the refinery does not.

### `InputError`

```python
class InputError(EngineException): ...
```

A call the library refuses: an empty scope, a workspace of another scope, tags outside their limits, runtime questions that do not fit or that the run already asks, `questions` given to a generation agent, a `task` or a `skill` given to a decision agent, a decision run with nothing to ask, media a decision model does not declare, a media part with both or neither of `data` and `url`, a correction of the wrong shape, a tool called with bad arguments or a path outside its directory.

### `ModelError`

```python
class ModelError(EngineException): ...
```

An adapter broke the contract: a result whose fields do not fit, a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels. Or a script did: a folder tool's result that does not match its `output_schema`, or a question's script that returns anything but an object or fills a question that does not fit.

### `RunawayError`

```python
class RunawayError(EngineException): ...
```

A generation run passed `max_steps` model calls without ending.

### `DelegationLoopError`

```python
class DelegationLoopError(EngineException): ...
```

A route would send its input to an agent that already decided it in the same chain of routes. It raises.

### `DelegationDepthError`

```python
class DelegationDepthError(EngineException): ...
```

Nesting went deeper than `max_depth`. Through `Delegate` it reaches the model as a tool error; through a route it raises.

### `StorageError`

```python
class StorageError(EngineException): ...
```

The storage refused: the database failed, or a row's scope is not the scope of the call.

### `WorkerError`

```python
class WorkerError(EngineException): ...
```

A tool or a question's script that runs in a worker process failed there. Its message is what the script raised, `ValueError: no order A-1` — `SystemExit: 3` for a `sys.exit(3)` — or what its import raised, a question's script's after the script and the question it fills (`engine: order.py of decision agent 'matcher' could not fill question 'order': KeyError: 'customer'`); a result JSON cannot hold, `TypeError: Object of type datetime is not JSON serializable`; a worker that stopped while the call was in flight — it exited (`workers/pool: the worker for lookup_order stopped: exited with code 3`), a signal ended it (`killed by signal 9`), the next call retired it once every thread was held past its deadline (`every thread was held by a call past its deadline`), or a call came from another event loop (`a call came from another event loop`); or a call in flight when `Engine.close` stopped its worker, `workers/pool: the worker for lookup_order stopped: it was closed`. Inside a run, the model reads a tool's as a tool error; a question's script's ends its decision run, before the decision model is called.
