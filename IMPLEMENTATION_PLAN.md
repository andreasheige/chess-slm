# ChessSLM — Implementation Plan

> A learning-first project for building a small chess language/model system from first principles, with the explicit goal of developing practical ML/DL engineering skills.

## 1. Purpose

ChessSLM is not primarily a "build the strongest chess engine" project.

The primary goal is to learn, implement, observe, and document the full lifecycle of a machine-learning system:

```text
raw data
  ↓
dataset construction
  ↓
representation / tokenization
  ↓
tensors
  ↓
model architecture
  ↓
forward pass
  ↓
loss
  ↓
backpropagation
  ↓
optimization
  ↓
evaluation
  ↓
experimentation
  ↓
serving
  ↓
MLOps
```

The project should remain understandable at every stage.

A successful implementation is one where we can explain:

- what every major tensor represents,
- why each architectural decision was made,
- what the model is optimizing,
- why training succeeds or fails,
- how we measure improvement,
- what trade-offs were introduced,
- and how the system moves from experiment to production.

---

# 2. Core Principles

## 2.1 Learning before abstraction

We prefer explicit implementations before high-level frameworks.

Example:

```text
First:
custom PyTorch training loop

Later:
Trainer / Accelerate / distributed frameworks
```

We should understand the mechanism before introducing abstractions.

---

## 2.2 Smallest possible iteration

Every implementation phase should produce something observable.

Avoid large jumps such as:

```text
"Build the transformer"
```

Prefer:

```text
1. encode one chess board
2. inspect the tensor
3. batch multiple boards
4. create embeddings
5. run one forward pass
6. inspect logits
7. calculate one loss
8. run backward()
9. inspect gradients
10. update weights
```

---

## 2.3 Prove correctness before scale

Before increasing:

- dataset size,
- model size,
- GPU count,
- training duration,
- infrastructure complexity,

we first prove that the smallest version works.

A key early milestone is intentionally overfitting a tiny dataset.

If the model cannot memorize a tiny dataset, scaling is not allowed.

---

## 2.4 No hidden magic

Whenever we introduce a new abstraction, library, or framework, we document:

```text
What problem does this solve?
What did we do before?
What is now abstracted away?
What do we lose visibility into?
```

---

## 2.5 Reproducibility

Every meaningful experiment should be reproducible from:

```text
code commit
+ config
+ seed
+ dataset version
+ environment
```

---

# 3. Guardrails

These guardrails apply throughout the project.

## G1 — No premature framework adoption

Do not introduce high-level ML tooling until the underlying mechanism has been implemented and understood.

Examples that should come later:

- Hugging Face Trainer
- Lightning
- Accelerate
- DeepSpeed
- PEFT
- LoRA
- experiment platforms
- orchestration platforms

---

## G2 — No unexplained generated code

Codex may generate code, but generated code must be understandable.

Before accepting non-trivial generated ML code, we should be able to answer:

```text
What enters this function?
What leaves it?
What are the tensor shapes?
What operation is performed?
Why is it necessary?
```

---

## G3 — Tensor shapes are documentation

For important tensors, document shapes explicitly.

Example:

```python
# board_tokens: [batch_size, 64]
# embeddings:   [batch_size, 64, embedding_dim]
# logits:       [batch_size, num_moves]
```

Shape bugs should never be "fixed until it runs" without understanding the mismatch.

---

## G4 — One conceptual leap at a time

Do not introduce several new ML concepts in the same implementation step unless unavoidable.

Bad:

```text
new tokenizer
+ transformer
+ scheduler
+ mixed precision
+ distributed training
```

Good:

```text
new tokenizer
→ verify

transformer block
→ verify

training loop
→ verify

scheduler
→ verify
```

---

## G5 — No scaling without a baseline

Before scaling anything, record a baseline.

Example:

```text
model: MLP-v0
dataset: 10k positions
top-1 accuracy: X
loss: Y
training time: Z
```

---

## G6 — Experiments must answer a question

Every experiment should begin with a hypothesis.

Example:

```text
Hypothesis:
Adding positional information for board squares should improve move prediction.

Change:
Add learned square embeddings.

Expected signal:
Lower validation loss / higher move accuracy.
```

---

## G7 — Failed experiments are first-class artifacts

Failures are not deleted from project history.

We record:

```text
what we expected
what happened
why we think it happened
what we changed next
```

These are especially valuable for future blog material.

---

## G8 — Stockfish is initially an evaluator, not a crutch

Early models must learn from data.

Do not build a system where the model simply delegates move selection to Stockfish.

Later Stockfish can be used for:

- evaluation,
- labeling,
- teacher/student distillation,
- generating value targets.

---

# 4. Repository Structure

Initial proposed structure:

```text
chess-slm/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── src/
│   └── chess_slm/
│       ├── __init__.py
│       ├── data/
│       ├── representation/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── utils/
│
├── scripts/
│   ├── data/
│   ├── training/
│   └── evaluation/
│
├── tests/
│
├── configs/
│
├── experiments/
│
├── notes/
│   ├── README.md
│   ├── decisions/
│   ├── experiments/
│   ├── failures/
│   └── concepts/
│
├── media/
│   ├── screenshots/
│   ├── diagrams/
│   └── plots/
│
└── artifacts/
```

Generated datasets, checkpoints, and large binary artifacts should not be committed directly to Git.

---

# 5. Documentation Trail

One goal of the project is to preserve enough context that the work can later become:

- a technical blog series,
- portfolio material,
- talks,
- internal presentations,
- learning notes.

We therefore maintain a lightweight engineering journal.

## 5.1 Daily / session notes

Create:

```text
notes/YYYY-MM-DD-session-N.md
```

Suggested template:

```markdown
# Session

## Goal

What am I trying to achieve?

## What I expected

My mental model before implementing it.

## What I changed

Files / architecture / data changes.

## Commands

Important commands used.

## Observations

What happened?

## Problems

Errors, unexpected behavior, wrong assumptions.

## What I learned

Concepts that became clearer.

## Open questions

Things I still do not understand.

## Next step

Exactly one next logical step.

## Blog material

Interesting screenshots, quotes, failures, diagrams, or insights.
```

---

## 5.2 Architecture decisions

Use lightweight ADRs:

```text
notes/decisions/ADR-0001-board-representation.md
```

Template:

```markdown
# ADR-XXXX: Decision

## Context

## Options

## Decision

## Why

## Consequences

## Revisit when
```

---

## 5.3 Experiment notes

Each meaningful experiment gets:

```text
experiments/EXP-XXXX-name/
```

Containing:

```text
config.yaml
README.md
metrics.json
```

Experiment README:

```markdown
# EXP-XXXX

## Question

## Hypothesis

## Baseline

## Change

## Result

## Interpretation

## Next experiment
```

---

## 5.4 Failure log

Useful failures get their own short notes:

```text
notes/failures/FAIL-XXXX-description.md
```

Capture:

```text
symptom
incorrect assumption
root cause
fix
lesson
```

---

## 5.5 Screenshot convention

Store meaningful screenshots under:

```text
media/screenshots/
```

Naming:

```text
YYYY-MM-DD_topic_short-description.png
```

Example:

```text
2026-10-03_training_first-loss-drop.png
2026-10-03_debug_illegal-move-encoding.png
```

Every screenshot worth keeping should be referenced from a note.

---

# 6. Implementation Roadmap

Each phase should be small enough to understand independently.

---

# Phase 0 — Project Bootstrap

## Goal

Create a clean, reproducible development environment.

### 0.1 Create repository

Deliverable:

```text
git repository
README.md
IMPLEMENTATION_PLAN.md
```

### 0.2 Select Python version

Record the decision.

### 0.3 Configure package management

Prefer a modern reproducible Python project setup.

### 0.4 Add PyTorch

Verify:

```python
import torch
print(torch.__version__)
```

### 0.5 Detect available compute

Inspect:

```text
CPU
CUDA / MPS
device
```

### 0.6 Add basic linting / formatting

### 0.7 Add test framework

### 0.8 Add first CI workflow

Initially:

```text
install
lint
test
```

### Exit criteria

```text
fresh clone
→ install
→ test
→ import project
```

works without manual fixes.

---

# Phase 1 — Understand the Raw Chess Data

## Goal

Understand the source data before building ML abstractions.

### 1.1 Obtain a very small PGN file

Do not start with millions of games.

Start with approximately:

```text
10–100 games
```

### 1.2 Read one game

Inspect:

- headers,
- moves,
- result,
- move numbering,
- SAN notation.

### 1.3 Parse one game programmatically

### 1.4 Replay moves on a board

### 1.5 Print board state after every move

### 1.6 Extract FEN after every move

### 1.7 Create first position → move pair

Conceptually:

```text
position_t → move_t
```

### Exit criteria

We can explain exactly how one PGN game becomes supervised training examples.

---

# Phase 2 — Define the Learning Problem

## Goal

Define precisely what the first model should learn.

Initial task:

```text
Given a chess position,
predict the move played in the dataset.
```

### 2.1 Define input

Candidate:

```text
board position
```

### 2.2 Define target

Candidate:

```text
played move
```

### 2.3 Decide whether metadata is included

Examples:

- side to move,
- castling rights,
- en passant,
- move counters.

### 2.4 Write ADR

```text
ADR-0001-initial-learning-objective.md
```

### Exit criteria

The ML problem can be described in one precise sentence.

---

# Phase 3 — Board Representation v0

## Goal

Convert a chess position into deterministic numeric data.

### 3.1 Inspect board squares

Map:

```text
a1 ... h8
```

to stable indices.

### 3.2 Define piece encoding

Example conceptual vocabulary:

```text
empty
white pawn
white knight
...
black king
```

### 3.3 Encode one board

Expected conceptual shape:

```text
[64]
```

### 3.4 Decode representation back into human-readable form

### 3.5 Unit-test round trips

### 3.6 Inspect multiple positions manually

### Exit criteria

We trust that:

```text
chess board → tensor
```

is deterministic and correct.

---

# Phase 4 — Move Representation v0

## Goal

Represent prediction targets numerically.

This phase should deliberately compare possible approaches before choosing one.

Potential representations:

```text
SAN vocabulary
UCI vocabulary
from-square × to-square
from/to/promotion structure
```

### 4.1 Collect unique moves from tiny dataset

### 4.2 Inspect vocabulary

### 4.3 Evaluate trade-offs

### 4.4 Choose v0 representation

### 4.5 Encode move

### 4.6 Decode move

### 4.7 Unit-test legal examples

### 4.8 Write ADR

```text
ADR-0002-move-representation.md
```

### Exit criteria

```text
move → target id → move
```

works reliably.

---

# Phase 5 — Dataset v0

## Goal

Create the first PyTorch-compatible dataset.

### 5.1 Convert PGNs into examples

Each example contains at least:

```text
board
target_move
```

### 5.2 Create Dataset class

### 5.3 Inspect one item

### 5.4 Create DataLoader

### 5.5 Inspect one batch

Explicitly document shapes.

Example:

```text
boards: [B, 64]
moves:  [B]
```

### 5.6 Train / validation split

### 5.7 Verify no accidental leakage

### Exit criteria

A DataLoader produces correct batches that we understand.

---

# Phase 6 — First Neural Network

## Goal

Build the simplest model that can learn anything.

Not a Transformer.

Possible first architecture:

```text
board tokens
→ embeddings
→ flatten / pooling
→ linear layers
→ move logits
```

### 6.1 Implement embedding layer

### 6.2 Run one board through embedding

Inspect shape.

### 6.3 Add first linear layer

### 6.4 Produce logits

### 6.5 Inspect logits

### 6.6 Convert logits to prediction

### Exit criteria

One batch can perform:

```text
input → model → logits → predicted move
```

---

# Phase 7 — Understand Loss

## Goal

Understand exactly what the model is optimizing.

### 7.1 Calculate cross-entropy manually on a tiny example

### 7.2 Use PyTorch cross entropy

### 7.3 Compare intuition with implementation

### 7.4 Inspect loss before training

### Exit criteria

We can explain what the loss value represents and why minimizing it helps.

---

# Phase 8 — Backpropagation

## Goal

Observe learning mechanics directly.

### 8.1 Run forward pass

### 8.2 Calculate loss

### 8.3 Call backward

### 8.4 Inspect gradients

### 8.5 Verify parameters have gradients

### 8.6 Perform one optimizer step

### 8.7 Compare parameters before / after

### Exit criteria

We can trace:

```text
prediction
→ error
→ gradient
→ parameter update
```

---

# Phase 9 — Minimal Training Loop

## Goal

Build training without high-level frameworks.

### 9.1 Loop over batches

### 9.2 Zero gradients

### 9.3 Forward

### 9.4 Loss

### 9.5 Backward

### 9.6 Optimizer step

### 9.7 Track mean loss

### 9.8 Add validation loop

### 9.9 Plot training / validation loss

### Exit criteria

Training runs for multiple epochs and produces metrics.

---

# Phase 10 — Intentionally Overfit Tiny Data

## Goal

Prove the pipeline can learn.

Dataset:

```text
~100–1,000 examples
```

Train until the model nearly memorizes it.

Observe:

```text
training loss → very low
training accuracy → very high
```

If this does not happen:

```text
STOP.
Do not scale.
Debug.
```

Possible causes:

- incorrect targets,
- representation bugs,
- optimizer issues,
- broken gradients,
- insufficient model capacity.

### Exit criteria

The model can deliberately overfit tiny data.

This is the first major milestone.

---

# Phase 11 — Establish Baseline v0

## Goal

Create the first honest benchmark.

Increase dataset modestly.

Track at minimum:

```text
training loss
validation loss
top-1 move accuracy
top-k move accuracy
illegal move rate
training duration
model parameter count
```

### Deliverable

```text
EXP-0001-baseline-v0
```

### Exit criteria

We have a reproducible baseline against which future models can be compared.

---

# Phase 12 — Data Pipeline v1

## Goal

Move from toy data toward realistic data volumes.

Incrementally scale:

```text
100 games
1k games
10k games
100k games
...
```

Study:

- preprocessing throughput,
- storage size,
- loading performance,
- shuffle strategy,
- memory usage.

Do not jump immediately to full public chess archives.

---

# Phase 13 — Better Position Representation

Potential experiments:

```text
piece embeddings
square embeddings
side-to-move embedding
castling representation
en-passant representation
```

Each change should be a separate experiment whenever possible.

---

# Phase 14 — Attention From Scratch

## Goal

Understand attention before using a full Transformer.

### 14.1 Create toy token sequence

### 14.2 Implement Q, K, V projections

### 14.3 Calculate attention scores

### 14.4 Scale scores

### 14.5 Softmax

### 14.6 Weighted value combination

### 14.7 Inspect attention matrix

### 14.8 Add multiple heads

### Exit criteria

We can explain and implement multi-head attention.

---

# Phase 15 — Transformer Block From Scratch

Implement incrementally:

```text
attention
residual
normalization
MLP
residual
```

Test every component separately.

---

# Phase 16 — Chess Transformer v1

Replace the simple baseline model with a small Transformer.

Start intentionally small.

Example order of magnitude:

```text
2–4 layers
small embedding dimension
few attention heads
```

Do not optimize for strength yet.

Compare against baseline.

---

# Phase 17 — Autoregressive Chess Model

## Goal

Move from board classifier toward language-model-style training.

Represent games as sequences.

Conceptually:

```text
<BOS> e4 e5 Nf3 Nc6 ... <EOS>
```

Train:

```text
token_t → token_t+1
```

Learn:

- causal masking,
- sequence batching,
- autoregressive loss,
- generation.

This is the transition toward a real ChessSLM.

---

# Phase 18 — Build a Small SLM

Scale the Transformer deliberately.

Potential progression:

```text
1M params
5M params
10M params
30M params
```

Study relationships between:

```text
model size
dataset size
compute
loss
playing strength
```

---

# Phase 19 — Stockfish Evaluation

Introduce Stockfish as an external evaluator.

Measure:

```text
legal move rate
best-move agreement
top-k agreement
centipawn loss
position evaluation difference
```

Do not yet train directly on Stockfish outputs.

---

# Phase 20 — Stockfish Distillation

Generate teacher labels:

```text
position
best move
top candidate moves
evaluation
```

Train student model against teacher signal.

Study:

```text
hard labels
soft targets
policy targets
value targets
```

---

# Phase 21 — Self Play

Introduce model-vs-model play.

Track:

```text
wins
losses
draws
illegal moves
game length
rating estimates
```

---

# Phase 22 — Reinforcement Learning

Only after supervised training is well understood.

Possible progression:

```text
reward modeling concepts
policy optimization concepts
self-play reward
Stockfish-derived dense rewards
```

Avoid jumping directly into RL frameworks.

---

# Phase 23 — Fine-tune an Existing Open SLM

Now compare our from-scratch knowledge with modern tooling.

Learn:

```text
Hugging Face Transformers
tokenizers
datasets
SFT
PEFT
LoRA
QLoRA
quantization
```

This phase intentionally comes late.

---

# Phase 24 — Serving

Expose inference.

Potential stack:

```text
model
↓
Python inference layer
↓
FastAPI
↓
web client
```

Measure:

```text
latency
throughput
memory
startup time
```

---

# Phase 25 — MLOps

Build production-oriented workflows.

Potential topics:

```text
model registry
dataset versioning
experiment tracking
artifact storage
CI/CD
GPU runners
container images
observability
reproducible deployment
```

This phase can leverage existing DevOps experience heavily.

---

# 7. Milestones

## M0 — Environment works

```text
repo
Python
PyTorch
tests
CI
```

## M1 — First training example

```text
PGN → board tensor + move target
```

## M2 — First neural prediction

```text
tensor → network → logits
```

## M3 — First learning

```text
loss decreases
```

## M4 — Tiny dataset memorized

```text
pipeline proven
```

## M5 — Baseline benchmark

```text
reproducible experiment
```

## M6 — Attention implemented

```text
attention understood from first principles
```

## M7 — Transformer trained

```text
Chess Transformer v1
```

## M8 — ChessSLM

```text
autoregressive model trained from scratch
```

## M9 — Engine evaluation

```text
Stockfish benchmark
```

## M10 — Distillation

```text
teacher → student learning
```

## M11 — Self-play / RL

```text
model improves through interaction
```

## M12 — Production system

```text
model served and observable
```

---

# 8. Codex Working Agreement

Codex should act as a pair programmer, not an autopilot.

For each non-trivial change:

1. Explain the intended change.
2. Keep the change as small as possible.
3. State expected tensor shapes when ML tensors are involved.
4. Add or update tests where appropriate.
5. Do not introduce new dependencies without explaining why.
6. Do not refactor unrelated code.
7. Prefer explicit code over clever abstractions while learning.
8. Stop at phase boundaries.
9. Record surprising failures or discoveries in notes.
10. Never silently change the learning objective.

Suggested repository instruction:

```text
This is a learning-first ML project.

Do not optimize for implementation speed at the expense of understanding.

When implementing ML functionality:
- explain tensor shapes,
- explain the mathematical operation,
- make one conceptual change at a time,
- avoid high-level abstractions until explicitly requested,
- include tests for representations and transformations,
- do not scale the model or dataset until the current stage is validated.

When a bug occurs, identify the underlying incorrect assumption before applying a fix.
```

---

# 9. Definition of Done for Every Phase

A phase is complete only when:

- the code runs,
- relevant tests pass,
- the result has been manually inspected where appropriate,
- the implementation is understood,
- notable decisions are documented,
- unexpected failures are recorded,
- there is a clear next step.

---

# 10. Immediate Next Step

Do **not** begin model implementation yet.

Next phase:

```text
Phase 0 — Project Bootstrap
```

The first implementation session should stop when:

```text
fresh clone
→ install dependencies
→ run tests
→ import torch
→ detect device
→ import chess_slm
```

works reliably.

Only then proceed to PGN data.
