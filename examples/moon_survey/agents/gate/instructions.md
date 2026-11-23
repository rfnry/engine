---
description: Says whether a tool call fits the survey, before the call runs
---
The app asks this agent about every tool call of the survey before the call runs, and runs the call only on a clear
yes: a `fits` probability of at least 0.5. An error, a timeout or a missing answer refuses the call.

A decision model is sent no prompt, so what the model reads, the state it judges and the question it answers, is in
`questions/fits.json`.
