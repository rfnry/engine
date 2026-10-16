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

A model that writes text and calls tools: any object with this method, matched by shape. It returns a `Reply`, or anything with its fields; a result whose fields do not fit makes the run raise `ModelError`. An adapter may also carry `media_types`, the file types its model takes, such as `{"image/*", "application/pdf"}`, so `Load` refuses the rest, and `model`, the name of the model it calls, a `str` the run records as its model when no reply reports one.

### `StreamingGenerationModel`

```python
@runtime_checkable
class StreamingGenerationModel(Protocol):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> ReplyLike: ...
```

A generation model that also streams. When a run has `on_event`, the engine calls `generate_stream` instead of `generate` and the adapter awaits `on_delta` with each piece of text as it arrives; it returns the whole reply, as `generate` does.

### `DecisionModel`

```python
@runtime_checkable
class DecisionModel(Protocol):
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

A model that answers typed questions about a state in one call: any object with this method, matched by shape. It returns an answer per question id, or a `Decided` when it can report what the call cost and which model answered. An adapter may also carry `model`, the name of the model it calls, a `str` the run records as its model when the call reports none, and `media_types`, the media its model reads in a state, such as `{"image/*"}`: a run's parts with that media reach it as the state, a list of parts; without it the state is text, and parts it does not declare are refused before it is called. Answers that do not fit — a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels — make the run raise `ModelError`.

### `ReplyLike`

```python
class ReplyLike(Protocol):
    text: str
    tool_calls: Sequence[ToolCallLike]
    usage: UsageLike | None
    stop_reason: StopReason
    model: str | None
    reasoning: Any
```

The shape a generation model's result is read by: the read-only fields of `Reply`. `ToolCallLike` is a protocol of `id: str`, `name: str` and `args: dict[str, Any]`, `UsageLike` is one of `input`, `output`, `cache_creation` and `cache_read`, each an `int`, and `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `DecidedLike`

```python
class DecidedLike(Protocol):
    answers: Mapping[str, AnswerLike]
    usage: UsageLike | None
    model: str | None
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
```

One turn of a conversation. An assistant turn carries its text in `content`, its calls in `tool_calls` and its `reasoning` as the model returned it; a tool result has `role="tool"`, the `tool_call_id` it answers, and `is_error` when the call failed.

`prefix` and `usage` are the engine's record of an assistant message it made from a reply, never sent to a model and outside the vocabulary shared with rfnry-memory: `prefix` is the first 16 hex characters of a sha256 over everything sent before the message when it was written — the model, as its adapter names it, the system prompt, the tools, the preloaded files and the earlier messages as sent — and its `reasoning` is sent again only while what precedes it is the same; `usage` is what the call that wrote it reported. A `usage` or `prefix` that does not fit these fields — another library's usage, a prefix that is not 16 hex characters — reads as `None`, so a message from elsewhere is taken without them; only a `prefix` that is dropped costs its `reasoning`, which is not sent again.

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
```

What `decide` returns instead of the answers alone when the adapter can report what the call cost and which model answered. The run's record carries both.

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
        name: str | None = None,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        tools: list[Tool] | None = None,
        refinery: Recorder | None = None,
        max_steps: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
        history_walk: int = 10_000,
        history_chars: int = 100_000,
    ) -> None: ...

    agents: dict[str, AgentInfo]
```

Loads the tree of agents under `root` — `root` itself when it has an `AGENT.md`, else its children that have one, else `root` as one plain generation agent named `name` or the folder's name — and checks it against the models and tool bodies given — the files of an agent with `preload: true` against the generation model's `media_types`; `tools` is one list for the whole tree, and `Recorder` is a protocol of `async def record(self, run: Run) -> None`, which a `Refinery` meets. `tool_timeout` is each tool call's deadline in seconds, `None` for none. `history_walk` is how many messages one `History` search walks back, `history_chars` how many characters one `History` result holds. `agents` maps each agent's name to its `AgentInfo`.

Raises `ConfigurationError` when `root` is not a directory, `name` given for a root with agents, or empty, when the tree cannot be used (a bad `AGENT.md`, question or tool file, an unknown frontmatter key, frontmatter that is not valid YAML or a `description:` that is not a string in a `.md` file the map would show, an agent in `agents:` or `route:` the tree does not have, a route entry naming a task its agent does not have or reaching a decision agent with nothing to ask — no `questions/` and no task with questions named — a route not by a choice question of the router's `questions/`, a decision agent with a file straight in `tasks/` or a task that repeats an id of its `questions/`, a decision agent with `preload: true`, a `reads:` naming a folder that is not a direct child of the root, starts with `.` or is the agent's own, `reads:` on a decision agent, `reads:` on an agent whose folder is a link to a folder elsewhere, a file `preload: true` would send that neither `Read` nor `Load` would hand over: over 20 MB, of a type the generation model's `media_types` does not name, or not the type its name says), when a tool is given that no agent declares or, with a generation model, a declared tool has no body, when two tools share a name, or when `max_steps`, `max_depth`, `tool_timeout`, `history_walk` or `history_chars` is not positive.

### `Engine.run`

```python
async def run(
    self,
    agent: str,
    scope: str,
    input: State,
    *,
    task: str | None = None,
    questions: Mapping[str, QuestionLike] | None = None,
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

Runs `agent` for the tenant `scope` on `input` and returns its record: a `Generation` for a generation agent, a `Decision` for a decision agent. A generation agent's `task` puts `tasks/<task>.md` in its system prompt; a decision agent's adds the questions in `tasks/<task>/` to those of `questions/`, which every run asks. `questions` are a decision agent's runtime questions, read by their fields and asked with the folder's in the same call. `history` is a list of messages or a `HistoryLike`, and the run sends its messages from position `start` on — `start=0` sends them all; a routed run is given both, a delegated run neither. A route's run gets the task its route entry names, if any, never the router's task or runtime questions. With a refinery attached, a run is recorded unless `record=False` — one that raises after its first model call too, as far as it got and with its `error`, before the exception reaches you — and a recording that fails is logged, never raised. `EventHandler` is `Callable[[Event], Awaitable[None]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets, and `tools` rides this run only; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`, called with the agent's name and the call.

Cancelling the task that runs it cancels the model call in flight and every tool call still running, a `Delegate` call and the run it started included, and re-raises the cancellation once all of them have ended, as `asyncio.TaskGroup` does; a tool that ignores cancellation holds it. Each further cancel meanwhile, two that land at the same moment counted as two, does not end the wait: it cancels once more everything still running, so a workspace command still being stopped is killed at once, with no grace period.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, a generation agent is given `questions`, a runtime question does not fit the contract or has an empty id or one the run already asks from `questions/` or its task, a decision run has no question to ask, a decision agent is given media its model does not declare in `media_types`, `tags` are outside their limits, `history` does not fit the contract or breaks a tool call from its result (a result whose call is not in it, or a call with no result), `start` is negative, past the end of a list `history`, or given without a `history`, or `input` mixes parts with other values. Raises `NotFoundError` when the tree has no agent `agent`, or the agent has no task `task`, before any model is called. Raises `ConfigurationError` when the model for the agent's kind is absent, a decision agent routes to a generation agent without a generation model, an agent with `workspace: true` runs without a `workspace`, `tools` repeats a name the agent already has, a tool it declares or `tools` holds is named `History`, a tool named `Output` collides with `expect`, or `approve` returns anything but `True` or `False`. Raises `ModelError` when an adapter's result does not fit the contract. Raises `RunawayError` when the loop passes `max_steps`. Raises `DelegationLoopError` when a route reaches an agent already in the same chain of routes, and `DelegationDepthError` when a route goes deeper than `max_depth`.

### `Engine.tool`

```python
def tool(
    self,
    agent: str,
    *,
    scope: str,
    name: str | None = None,
    description: str | None = None,
    context: str | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
) -> Tools: ...
```

Returns a `Tools` of one tool that runs `agent` as a top-level run under `scope`, recorded like any run: named `name` or the agent's name, described by `description` or the agent's, taking `{"input": str}` — with an optional `task` when the agent has tasks and optional `questions` when it is a decision agent, as `Delegate` takes them — and returning a generation agent's reply or a decision agent's answers as a dict. `context`, `tools`, `workspace`, `tags` and `approve` ride every call. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, or `tags` are outside their limits. Raises `NotFoundError` when the tree has no agent `agent`. A call raises what `Engine.run` raises.

### `Stop`

```python
class Stop:
    def __init__(self) -> None: ...
```

A person's handle on a running run, passed as `run(stop=)`; it reaches the runs it delegates and routes to. A stop set before the run starts ends it before the first model call; one set after it ended does nothing.

### `Stop.now`

```python
def now(self) -> None: ...
```

Cancels the model call in flight and every tool call still running, except a running `Delegate` call, which is waited for while the stop reaches the run it started; a call it finds not yet running never runs, so a stop from an `on_event` handler of a call's `ToolCallStarted`, or of a reply's `TextDelta`, comes before the tool runs. A handler still running when a stop cancels its call, or a streamed `TextDelta` handler, is cancelled with that work, so a kill switch does its own work, such as an audit, before it calls `stop.now()`. The run returns with `stopped="now"`, and each call it cut off is answered with the error `stopped by the user`. A call that already has its result keeps it, and the stop waits for an `on_event` still handling its `ToolResultReady` to return, so a slow handler delays it; cancelling the run cancels the handler. The run returns once the work the stop cancelled has ended, so a tool that ignores cancellation holds the stop, as it holds a cancel of the run. `now()` is safe to call any number of times: each call after the first, while the run is stopping, cancels once more every tool call still running — not a running `Delegate` call, whose run takes that `now()` itself, nor a call whose `ToolResultReady` handler the stop waits for — so a workspace command still being stopped is killed at once, with no grace period, as it is when the stopping run is cancelled.

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
    kind: Kind
    description: str
    agents: list[str]
    reads: list[str]
```

An agent as the engine loaded it, in `Engine.agents`: its name, its kind, its description, the agents it may call and the folders of its root it reads. `Kind` is `Literal["generation", "decision"]`.

### `Generation`

```python
class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    task: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
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

The record of a generation run: the `context` it was given, when it started (`at`), which agent folder and tools produced it (`fingerprint`), the system prompt as sent, the messages as sent to the first model call — the history, then the input — the whole `conversation`, the tools offered, every tool call as a `Step`, the model calls and their summed usage, the reply, and the runs it delegated to in `children`, in step order. `model` is the model the replies reported; when none did, as in a run stopped during its first model call, it is the adapter's `model` attribute if that is a non-empty string, else `None`. An accepted `Output` makes `reply` its JSON. `preload` is what `preload: true` sent, `[]` without it: the files sorted by relative path character by character, a file `Read` returns as a `Text` of `<<<path>>>` on a line of its own followed by what `Read` returns, any other file as a `Text` of `<<<path>>>` followed by the part `Load` returns; every model call receives it as one user message ahead of the messages, and neither `messages` nor `conversation` holds it. `Stopped` is `Literal["now", "after_step"]`; `stopped` is `None` for a run that ended on its own. `start` is the position in the history the run began at, `0` when it was sent whole. `added` is computed from the record: the input and every message after it, `conversation` from the input on — what a caller keeps. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`; `stop_reason` is the last reply's `stop_reason` — `"max_tokens"` for a reply the model's limit cut off, `"tool_use"` after an accepted `Output` — and `None` when no reply came back. `error` is `None` for a run that returned; a run that raised after its first model call is recorded with `error` as `"<Type>: <message>"`, `reply` `""` and what it had done by then.

### `Decision`

```python
class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    task: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
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

The record of a decision run: the task it was given, the state as the model was handed it — a list of parts when the adapter's `media_types` took them, read back as parts — every question asked, from the folder or the task and at run time, with `runtime` naming the ids that came at run time, an answer per question id with its `confidence`, the usage and model the adapter reported — its `model` attribute, a non-empty string, when the call reports no model — and what produced it (`fingerprint`, whose `tools` is the fingerprint of no tools and whose `questions` covers `questions`). `routed_to` is the run its route started, if any. `added` is the `added` of the run its route started, `[]` when it routed nowhere. `error` is set, and `answers` empty, when the decision model's call raised; a routed run that failed sits in `routed_to` with its own `error`, or `routed_to` is `None` when the routed run was refused before its first model call, and the router carries the same `error`.

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

Which agent, which tools and which questions produced a run, each the first 16 hex characters of a sha256. `agent` covers every file in the agent's directory and in each folder it names in `reads:` as the engine loaded them, by relative path and content, a read folder's files under `../<name>/`, leaving out hidden files and folders (a name starting with `.`) and links that lead out of their folder; it is computed once, when the `Engine` is built. `tools` covers the tools the model was offered, `Run.tools` sorted by name as canonical JSON: the folder's, the code's, the run's `tools=` and `Output` when `expect=` asked for one. `questions` covers the questions a decision run asked, `Decision.questions` as JSON ordered by question id, each question's fields in order and its options and levels as given, runtime questions included; it is `None` for a generation run. None covers the input, the history, `context`, `tags`, the model or the code behind a tool.

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

One tool call of a generation run: the tool, its arguments, its result, whether it failed, when it finished and how long it took. `child_run_id` names the run a `Delegate` started, `stopped` marks a call `Stop.now()` cut off, and `approved` is what `approve` answered, `None` when nobody was asked.

### `TextDelta`

```python
class TextDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str
```

Text from the model, sent to `on_event` as it is produced; from a model that does not stream, the whole text of each reply. `turn` is the model call of the run that said it, from 0: text of a turn that ended in tool calls was said beside them, and a run that ends in text replies with its last turn's text. A delegated or routed run sends its own `run_id` and counts its own turns.

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

Sent to `on_event` once, when a tool call is done: `step` is the `Step` the run records for it, equal to its entry in `Generation.steps`, sent on every way a call ends — a result, an error, a timeout, a call `approve` refused, a delegation, a call `Stop.now()` cut off, whether or not it had started. Any other call the engine answers without running it — `Output`, a call after an accepted `Output`, a call in a reply that ended for a reason other than tool use, answered `NotRun` — sends neither `ToolCallStarted` nor `ToolResultReady`; its step is in `Generation.steps`. A cancelled run raises and is not recorded, so a call it cancels sends no `ToolResultReady`, a call whose tool returns after the cancel included: it ends with the cancellation before any handler sees its result.

### `Event`

```python
Event = TextDelta | ToolCallStarted | ToolResultReady
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
```

A stored label: its `source` and what it says — a person's `review`, the world's `outcome`, or a judge's or a teacher's `answers`.

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

One tool: a name, a description, an input schema and a body. `Body` is `Callable[[dict[str, Any]], Awaitable[Result]]`, an async function of the arguments, where `Result` is `str | dict[str, Any] | list[Part]`; `args` validates the arguments and gives the schema unless `input_schema` does. In `Engine(tools=)` it is the body behind a tool an agent declares, and a declaration with only a name adopts its description and schema.

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
    ) -> None: ...
```

The tools `agent` runs with under `scope`, as a `Tools` to drive without a model: the builtins, `History` when `start` is above 0, the tools it declares, the workspace's tools when it declares `workspace: true` or is a plain agent (a root with no `AGENT.md`) given a `workspace`, `Delegate` when it names `agents:`, and `tools`. `Delegate` takes `agent` and `input`, an optional `task` when one of the agents it lists has tasks, and optional `questions` — ids to objects with the fields of `Choice`, `Score` or `Check` — when one is a decision agent; its description lists each agent's name, description, kind and tasks. A `Delegate` dispatched through it runs the child as a top-level run with this scope, `tools`, `workspace`, `tags` and `approve`, and the `task` and `questions` it is called with, and raises what `Engine.run` raises; inside a run the same refusals reach the model as tool errors. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. `History` reads `history` before position `start`: a page of up to `limit` messages — 20, at most 100 — before `before`, which defaults to `start` and never passes it, each under a JSON header with its `position`, `role`, and its `tool_calls`, or `tool_call_id` and `is_error`, in the order they happened; with `pattern`, a regular expression (`case_insensitive`), only the messages whose text matches, walking back at most the engine's `history_walk` messages. An image, audio or document shows as a line naming it — its type, its media type, a document's `name` and its `url` when it has one — never as the file. A result holds at most `history_chars` characters of messages and, when there is more, ends with where to go on; the engine's `tool_timeout` bounds a search's pattern.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, `start` is negative, `start` is above 0 without a `history`, or `start` is past the end of a list `history`. Raises `NotFoundError` when the tree has no agent `agent`. Raises `ConfigurationError` when the agent declares `workspace: true` and no `workspace` is given, a tool it declares has no body, or two tools share a name. A tool it declares or `tools` holds named `History` is a `ConfigurationError` whatever `start` is: the name is the builtin's on every run.

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

Raises `ConfigurationError` when `backend` lacks `spawn`, `read` or `write`, its `description` is not a string, `scope` is empty, or `timeout` or `output_bytes` is not positive.

### `Workspace.tools`

```python
def tools(self) -> list[Tool]: ...
```

Returns the five tools that an agent with `workspace: true`, or a plain agent (a root with no `AGENT.md`), gets whenever a workspace is passed: `Shell(command, background=false)`, which returns `exit_code`, `stdout`, `stderr` and `timed_out`, or with `background=true` starts a job and returns `{"job": id}` at once; `Job(id, wait=0, stop=false)`, which waits up to `wait` seconds for the job to end, then returns its `output` since the last read, `running`, and `exit_code` once it has ended, and with `stop=true` stops a job still running after the wait; `Jobs()`, which lists the jobs as `{"jobs": [{job, command, started, running, exit_code}]}`; `Write(path, content)`, which writes a file exactly; and `Edit(path, old, new, all=false)`, which replaces one exact occurrence, or every occurrence with `all=true`, in a UTF-8 file of at most 5 MB, and says how many it replaced. They go through the backend, and its errors reach the caller as they are. `Job` raises `InputError` for a `wait` outside 0 to `timeout`, or an id with no job: a job leaves the list once its last output is read.

### `Workspace.close`

```python
async def close(self) -> None: ...
```

Stops every job of the workspace with its process's `kill()`, waits for each to end, and empties the list; a job whose `kill()` raised stays listed, so a second `close` or `Job(id, stop=true)` can reach it. The workspace stays usable and its ids keep counting. Leaving `async with Workspace(…) as workspace:` calls it. Raises the first error a `kill()` raised, after asking every job, and logs the others. A cancelled `close` returns once every `kill()` has returned — a second cancel cancels each `kill()` still running, once — and logs the `kill()` errors it cannot raise; cancelled while its kills run, it leaves the list as it is.

### `Backend`

```python
@runtime_checkable
class Backend(Protocol):
    async def spawn(self, command: str) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...
```

Where a workspace's commands run and its files live, passed to `Workspace`. `spawn` starts `command` in the workspace's directory with no input and returns at once, and must return promptly: a stop or a cancel that comes while it runs waits for it to return, then for the kill of what it started, however many cancels come meanwhile, and a second one cancels that kill as soon as it starts; with `background=true` it waits for the job to be listed, so `close` stops it. `read` returns a file's bytes and refuses one larger than `limit`; `write` writes a file whole, creating its parent directories. Paths are the model's, relative to the workspace, and what they can reach is the backend's to decide. An error a backend raises reaches the model as the tool's error. A backend may have a `description`, a string `Workspace` ends `Shell`'s description with.

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
        description: str | None = None,
    ) -> None: ...

    root: Path
    shell: list[str]
    env: dict[str, str]
    description: str | None

    async def spawn(self, command: str) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...
```

A `Backend` on this machine. Commands run as `[*shell, command]` in `root`, in a new session, with `env` as their whole environment — by default `PATH`, `HOME` set to `root`, and `LANG`; a sandbox goes in front of the shell, as in `shell=["bwrap", …, "bash", "-c"]`. `kill` sends the command's process group SIGTERM and, a second later or at once when it is cancelled, SIGKILL. `read` and `write` stay inside `root`, symlinks resolved, and each runs in a thread: a call cancelled meanwhile returns once its thread has ended, however many cancels come, so a `Write` or `Edit` cancelled while it writes has written its file when the cancel returns. `description` ends `Shell`'s description.

Raises `ConfigurationError` when `root` is not a directory, `shell` is not a non-empty list of non-empty strings, its first word is not on the `PATH` of `env`, or `description` is not a string. `spawn` raises `InputError` when the command cannot start, as when `root` was removed: `workspace: cannot start the command: No such file or directory`. `read` and `write` raise `InputError` for a path outside `root`, through a link that leads out of it or one swapped in while they walk to the file, and for every path once `root` is no longer the folder it was when `Local` was made — replaced, or reached through a link swapped in for it or a folder above it; `read` for a path that is not a file — missing, a directory, a FIFO or a socket, or a path through a file, such as `notes.txt/x` — or a file larger than `limit`; `write` for a directory, a FIFO or a socket, and a path through a file as `workspace: cannot write notes.txt/x: Not a directory`. Any other error the system gives is `workspace: cannot read {path}: {reason}` or `workspace: cannot write {path}: {reason}`. Every message names the path as the model gave it, never the path on this machine. The root check compares directory inode numbers, so a filesystem where they are not stable — overlayfs without `xino`, FUSE without `use_ino` — refuses a valid root and is not supported as a workspace root.

## Storage

### `Storage`

```python
@runtime_checkable
class Storage(Protocol):
    async def initialize(self) -> None: ...

    async def put_run(self, scope: str, row: RunRow) -> None: ...

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

The one storage protocol, as the refinery calls it; `PostgresStorage` implements exactly these methods, as a storage of your own does. Every method but `initialize` takes `scope` first and reads and writes that scope only: `initialize` creates what the storage needs and is safe to run again; `put_run` stores a row, replacing the one with the same id; `put_label` stores a row unless the scope already holds a label with its id, which it never replaces, and returns the row stored under that id, so calls racing on one id all get the same row; `get_run` returns a run or `None`; `runs` returns runs newest first by `at`, at most `limit` or all with `None`, narrowed by `run_ids`, `kind`, `since` (stored after it, by `recorded`) and `unlabeled_by`, which keeps the runs no label from that source names; `labels` returns labels oldest first, narrowed by `run_ids`, `source` and `since`; `run_tree` returns the runs `run_ids` or `tags` (by containment) match and every run under them, oldest first; `delete_runs` removes those runs and every label on them, and `delete_scope` every run and label in the scope, each returning the number of rows removed; `count_scope` counts the scope's runs and labels. A storage raises `StorageError` when a row's scope is not the call's, and `PostgresStorage` raises it whenever the database refuses.

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

A stored run. `body` is the run's record as JSON with its children and routed run left out, each a row of its own joined by `parent_run_id`; `at` is when the run started and `recorded` when the refinery stored it.

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

Raises `InputError` when the run's scope is empty.

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

Writes every run recorded or labeled after `since`, with all its labels, to `bucket` as JSONL under `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, in the datasets `generation`, `preference` and `decision`. Every row carries `holdout`, from a hash of its run id and `holdout_percent`, and its run's `fingerprint` as `{"agent", "tools", "questions"}`. A generation row carries the run's `stopped` and `stop_reason`. A run whose `error` is set is in no dataset, except a decision run that answered before the run it routed to raised: its rows are written. Returns the keys, the rows per dataset and the watermark for the next `since`.

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
) -> AsyncGenerator[Replay, None]: ...
```

Runs `candidate` on what the top-level runs of `agent` in `scope` were given and yields a `Replay` for each, oldest first by when the run started. `holdout` keeps the held-out runs, by the same hash and `holdout_percent` as `export`; `labeled` the runs with a `human` or `world` label; `tags` the runs whose tags contain these, as `delete_runs` matches; `fingerprint` the runs whose fingerprint equals it, all three parts; `limit` the first so many. Each replay is `candidate.run(agent, scope, …, record=False)` with the recorded input — parts read back as parts — the history (the messages sent before the input) with `start=0`, so without `History` — or, given `conversation`, what it returns for the recorded run, cut before the recorded input, with the recorded `start`, so the candidate has the same window and `History` over everything before it — the task and the `context`; a decision run replays its state, its task and its runtime questions as recorded, the rest of its questions coming from the candidate's folder, and when its route reached a generation run, that run's input, history and context. `expect`, `approve`, `tools` and `workspace` are passed to every replay. `conversation` is called once per replay, with the recorded top-level run, and applies to the generation run it replays; a decision run that reaches no generation run never calls it, and a `conversation` that raises, or that returns a conversation ending before the recorded input, gives the pair its `error`. It gives the recorded window only when the run was sent its conversation with `start`: a run recorded with `start=0` on a list the app had already cut replays on the conversation's first messages instead. `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. The candidate's tools run for real. Each run is replayed `repeat` times, each attempt a run of its own yielded as a pair of its own whose `attempt` counts from 0; the attempts of a run come one after another, in attempt order, and `limit` counts recorded runs, not attempts. At most `concurrency` replays run at once, attempts included, and while a pair is being read at most `concurrency - 1` run, so with the default a replay starts when its pair is asked for. Pairs are yielded in that order either way. Closing the generator, or cancelling its reader while it waits for a pair, cancels each replay still running and waits for them to end, and each further cancel meanwhile cancels them once more, so a second ends a replay's workspace command at once: the close then returns once they have ended, and the reader raises `CancelledError`. A candidate that raises yields its pair with `error` and the replay goes on, the other attempts of the same run included. Nothing is recorded and no label is written.

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

A call the library refuses: an empty scope, a workspace of another scope, tags outside their limits, runtime questions that do not fit or that the run already asks, `questions` given to a generation agent, a decision run with nothing to ask, media a decision model does not declare, a media part with both or neither of `data` and `url`, a correction of the wrong shape, a tool called with bad arguments or a path outside its directory.

### `ModelError`

```python
class ModelError(EngineException): ...
```

An adapter broke the contract: a result whose fields do not fit, a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels.

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
