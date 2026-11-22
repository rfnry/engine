---
description: Says whether a tool call fits the survey, before the call runs
---
The app asks this agent about every tool call of the survey before the call runs, and runs the call only on a clear
yes: a `fits` probability of at least 0.5. An error, a timeout or a missing answer refuses the call.

The state is the call: `agent`, the agent making it; `tool` and `args`; `run_id`, the run making it; and `region`, the
region that run holds on the board, `null` when it holds none.
