# ChessSLM — Agent Instructions

This is a learning-first ML/DL project.

The primary goal is not to build the strongest chess model as quickly as possible.
The primary goal is for the developer to understand how the system works from
raw data through training, evaluation, and eventually deployment.

## Core behaviour

- Work as a pair programmer, not an autopilot.
- Make one conceptual change at a time.
- Do not continue into the next implementation phase unless explicitly asked.
- Prefer simple and explicit implementations over clever abstractions.
- Do not introduce dependencies without explaining why they are needed.
- Do not refactor unrelated code.
- Understand bugs before fixing them.
- Never hide complexity merely to make something work.

## ML / DL guardrails

When implementing ML functionality:

- Explain what enters an operation and what leaves it.
- State important tensor shapes explicitly.
- Explain new mathematical operations before or while implementing them.
- Do not introduce high-level ML abstractions until the underlying mechanism
  has been implemented and understood.
- Do not scale datasets or models until the current implementation has been
  validated.
- Never silently change the learning objective, representation, loss function,
  or evaluation methodology.

## Debugging

When something fails:

1. Describe the observed behaviour.
2. State the expected behaviour.
3. Form a hypothesis about the underlying cause.
4. Verify the hypothesis where practical.
5. Apply the smallest reasonable fix.
6. Verify the fix.
7. Record noteworthy failures when they provide useful learning material.

Do not blindly modify parameters or code until an error disappears.

## Documentation

This project should preserve useful material for future technical writing.

When something noteworthy happens, suggest capturing it if appropriate:

- important design decisions,
- failed assumptions,
- useful errors,
- experiment results,
- training curves,
- interesting terminal output,
- screenshots,
- conceptual breakthroughs.

Do not turn every trivial action into documentation.

## Experiments

Every meaningful experiment should have:

- a question,
- a hypothesis,
- a baseline when applicable,
- one primary change,
- a result,
- an interpretation.

Prefer changing one meaningful variable at a time.

## Phase boundaries

Follow `IMPLEMENTATION_PLAN.md`.

Before starting work, identify the current phase and immediate objective.

At the end of a phase:

- verify the exit criteria,
- summarize what was learned,
- identify unresolved questions,
- stop before beginning the next phase unless explicitly instructed.