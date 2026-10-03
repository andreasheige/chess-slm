# ChessSLM — Build Log 03
## Making ML experiments resumable, reproducible, and comparable

**Date:** 2026-10-04  
**Status:** Experiment infrastructure established  
**Previous log:** `ChessSLM_BUILD_LOG_02.md`  
**Purpose:** Raw material for future blog posts / technical write-ups

---

# Where Build Log 02 ended

Build Log 02 ended with the first meaningful multi-game baseline:

```text
198 games total
158 training games
40 validation games
13,951 training positions
2,983 validation positions
```

The baseline learned the training distribution strongly, but unseen-game validation plateaued around 8–9% full-move accuracy.

That changed the question from:

> Can the model learn?

to:

> Can I run experiments reliably enough to prove whether the next model is actually better?

Build Log 03 is about building that infrastructure.

# 1. `latest.pt` and `best.pt` are different concepts

The first checkpoint system stored only `latest.pt`, which was enough to resume training.

Validation introduced another requirement: the latest model is not necessarily the model that generalizes best.

The checkpoint system was split conceptually:

```text
latest.pt
→ latest training state
→ used for resume

best.pt
→ best validation result
→ used for model selection
```

A real run demonstrated this:

```text
epoch 0   val_move=4.89% → new best
epoch 10  val_move=8.65% → new best
epoch 20  val_move=8.55%
epoch 30  val_move=8.62%
```

The best model and latest model can therefore point to different epochs.

# 2. Best validation history must survive resume

The first implementation initialized:

```python
best_val_move_acc = 0.0
```

for every process.

That is correct for a fresh experiment but wrong for resumed training. A new process must know the best score already achieved.

Checkpoint state was expanded to store:

```text
model_state_dict
optimizer_state_dict
epoch
best_val_move_acc
```

A clean run reached:

```text
epoch 20
best_val_move_acc = 8.68%
```

and saved that history.

A later process resumed from epoch 30 and correctly printed:

```text
Checkpoint loaded ... (epoch 30, best_val_move_acc=8.68%)
```

At epoch 40 validation was only 8.35%, so no new `best.pt` was written.

That verified model-selection history across process restarts.

# 3. Checkpoint schema evolution

Older checkpoints did not contain `best_val_move_acc`.

Loading them initially produced:

```text
KeyError: 'best_val_move_acc'
```

The loader became backward-compatible:

```python
checkpoint.get("best_val_move_acc", 0.0)
```

This allows older checkpoints to remain loadable even as the persisted schema evolves.

It cannot reconstruct history that was never saved, but it avoids crashing solely because a newer field was introduced.

# 4. A tiny Python bug caught immediately

The first fallback implementation accidentally used:

```python
checkpoint["best_val_move_acc", 0.0]
```

instead of `.get(...)`.

Python interpreted that as a tuple key and raised:

```text
KeyError: ('best_val_move_acc', 0.0)
```

The checkpoint tests caught it.

A recurring lesson in ChessSLM: experimental validity still depends on ordinary software correctness.

# 5. Experiment configuration became explicit

`run.py` still contained experiment-defining magic numbers:

```text
PGN path
batch size
epochs
learning rate
validation fraction
seed
logging cadence
checkpoint paths
```

A frozen dataclass was introduced:

```python
@dataclass(frozen=True)
class TrainingConfig: ...
```

It now describes values such as:

```text
experiment_name
pgn_path
batch_size
epochs
learning_rate
validation_fraction
seed
log_every
artifact_dir
```

The separation became:

```text
TrainingConfig
→ what experiment are we running?

run.py
→ how is an experiment executed?
```

`frozen=True` makes the experiment definition immutable after creation.

# 6. Tests should verify contracts, not temporary values

The first config test asserted:

```python
assert config.epochs == 40
```

When the experiment target intentionally changed to 50, the test failed even though nothing was broken.

The test was rewritten around actual invariants:

```text
batch_size > 0
epochs > 0
learning_rate > 0
0 < validation_fraction < 1
seed >= 0
log_every > 0
experiment_name is non-empty
paths have the expected type
```

A separate test verifies that config values can be overridden.

Principle:

```text
Bad test:
"epochs must be 40"

Better test:
"epochs must be valid"

Useful test:
"TrainingConfig can represent different experiments"
```

Experiment parameters are supposed to vary.

# 7. `run.py` stopped owning magic numbers

The training entrypoint was refactored to use the config:

```text
config.batch_size
config.learning_rate
config.epochs
config.validation_fraction
config.seed
config.log_every
```

Startup output now carries experiment context:

```text
Device: mps
Seed: 42
Batch size: 8
Learning rate: 0.001
Epoch target: 50
```

This makes individual runs easier to understand and compare.

# 8. Metrics became structured data

Previously, metrics primarily lived in terminal output.

A new immutable dataclass was introduced:

```python
@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    train_from_acc: float
    train_to_acc: float
    train_move_acc: float
    val_from_acc: float
    val_to_acc: float
    val_move_acc: float
```

At each evaluation point:

```text
evaluate training data
evaluate validation data
        ↓
EpochMetrics(...)
        ↓
history.append(...)
```

The learning curve now exists as structured data instead of only formatted strings.

# 9. Experiment artifacts

A persistence layer was added for:

```text
config.json
metrics.json
```

An example configuration:

```json
{
  "experiment_name": "baseline_v1",
  "pgn_path": "data/raw/MacKenzie.pgn",
  "batch_size": 8,
  "epochs": 10,
  "learning_rate": 0.001,
  "validation_fraction": 0.2,
  "seed": 42,
  "log_every": 10,
  "artifact_dir": "artifacts"
}
```

An example metrics history contains evaluation points such as:

```text
epoch 0
loss ≈ 7.63
train_move ≈ 5.33%
val_move ≈ 4.89%

epoch 10
loss ≈ 4.78
train_move ≈ 16.50%
val_move ≈ 8.68%
```

Terminal scrollback is no longer the only record of the experiment.

# 10. Global checkpoint paths were still dangerous

The first artifact structure looked like:

```text
artifacts/
└── baseline_v1/
    ├── config.json
    └── metrics.json

checkpoints/
├── latest.pt
└── best.pt
```

That still allowed one experiment to accidentally discover another experiment's checkpoint — a failure mode the project had already encountered.

The infrastructure should prevent that by design.

# 11. Experiments now own their checkpoints

Checkpoint paths were removed from `TrainingConfig`.

Instead, they are derived from experiment identity:

```python
experiment_dir = config.artifact_dir / config.experiment_name
latest_checkpoint_path = experiment_dir / "latest.pt"
best_checkpoint_path = experiment_dir / "best.pt"
```

The intended layout is now:

```text
artifacts/
└── baseline_v1/
    ├── config.json
    ├── metrics.json
    ├── latest.pt
    └── best.pt
```

A future `baseline_v2` automatically gets a separate directory and separate state.

Experiment identity now scopes:

```text
configuration
metrics
latest training state
best model
```

# 12. Another stale test exposed the intentional refactor

After checkpoint paths were removed from `TrainingConfig`, a test still asserted that those attributes existed.

The test failed with an `AttributeError`.

The implementation was correct; the test was stale.

Those assertions were removed because checkpoint paths are now derived from experiment identity rather than stored as independent config values.

# 13. Metrics history had the same resume problem

Model state survived resume.

Optimizer state survived resume.

Epoch survived resume.

Best validation score survived resume.

But metrics history initially did not.

Every new process still created:

```python
history = []
```

So a resumed process could overwrite an existing learning curve with only its new evaluation points.

Example of the bug:

```text
process A
→ metrics: [0, 10]

process B
→ history = []
→ epoch 20
→ save [20]
```

The earlier experiment history would disappear.

# 14. Metrics history became resumable

A new helper was added:

```python
load_metrics_history(...)
```

Behavior:

```text
metrics.json exists
→ deserialize JSON
→ reconstruct EpochMetrics
→ return existing history

metrics.json missing
→ return []
```

`run.py` now restores history before continuing training.

New evaluation points append to the previous learning curve.

# 15. End-to-end metrics resume verification

A clean experiment called:

```text
metrics_resume_test
```

was used to verify behavior across two processes.

First process:

```text
epoch 0
epoch 10
```

Second process resumed and added:

```text
epoch 20
```

Final `metrics.json` contained all three entries:

```text
0
10
20
```

with real values approximately:

```text
epoch 0
loss        7.6285
train_move  5.33%
val_move    4.89%

epoch 10
loss        4.7603
train_move 16.99%
val_move    8.55%

epoch 20
loss        3.9064
train_move 27.63%
val_move    8.38%
```

The learning curve now survives process boundaries.

# 16. The experiment is larger than one Python process

This is the central conceptual change of Build Log 03.

Earlier:

```text
training run ≈ one Python execution
```

Now:

```text
experiment
    │
    ├── process A
    │     train
    │     save
    │
    ├── process exits
    │
    ├── process B
    │     restore
    │     continue
    │     save
    │
    └── ...
```

while the experiment remains one continuous object:

```text
artifacts/<experiment_name>/
├── config.json
├── metrics.json
├── latest.pt
└── best.pt
```

The process is temporary.

The experiment persists.

# 17. Current artifact semantics

## `config.json`

Answers:

> What experiment did we intend to run?

It records dataset/configuration choices such as batch size, epoch target, learning rate, validation fraction, seed, and experiment identity.

## `metrics.json`

Answers:

> What happened during training?

It stores the learning curve across process restarts.

## `latest.pt`

Answers:

> Where can training continue from?

It contains model state, optimizer state, epoch, and best validation history.

## `best.pt`

Answers:

> Which checkpoint produced the best validation move accuracy so far?

These are four distinct responsibilities rather than four generic files.

# 18. Why this matters before changing the model

The baseline already established a significant generalization gap.

The next phase will likely change things such as:

```text
board representation
side-to-move information
legal move information
spatial modeling
hidden dimensions
move decoding
attention
```

Without experiment infrastructure, comparison becomes:

```text
"I think this run looked better."
```

With the current infrastructure:

```text
Experiment A
├── config
├── complete metrics
├── latest state
└── best model

vs

Experiment B
├── config
├── complete metrics
├── latest state
└── best model
```

Architecture work can now become controlled experimentation rather than terminal-memory comparison.

# Milestone reached — experiment infrastructure

ChessSLM now supports:

```text
✓ latest checkpoint
✓ best validation checkpoint
✓ best-score history across resume
✓ backward-compatible checkpoint loading
✓ immutable TrainingConfig
✓ configurable experiment parameters
✓ structured EpochMetrics
✓ JSON config artifacts
✓ JSON metrics artifacts
✓ experiment-scoped directories
✓ experiment-scoped checkpoints
✓ resumable metrics history
✓ resumable model state
✓ resumable optimizer state
✓ resumable epoch state
✓ resumable model-selection state
```

The experiment layout is:

```text
artifacts/
└── <experiment_name>/
    ├── config.json
    ├── metrics.json
    ├── latest.pt
    └── best.pt
```

# Failures worth preserving

## Best score reset on resume
Resume initially forgot previous validation history.

**Lesson:** resumable state includes more than weights.

## Old checkpoint missing a new field
Checkpoint schema evolution produced a `KeyError`.

**Lesson:** persisted ML state needs compatibility thinking.

## Incorrect dictionary fallback syntax
A tuple-key lookup accidentally replaced `.get()`.

**Lesson:** small Python bugs can invalidate experiment orchestration.

## Config test coupled to `epochs == 40`
A legitimate experiment change broke a brittle test.

**Lesson:** test contracts, not temporary choices.

## Global checkpoint paths
Experiments could accidentally share state.

**Lesson:** experiment identity should scope persisted files.

## Stale test after checkpoint-path refactor
Tests still expected fields intentionally removed from config.

**Lesson:** tests are part of the architecture.

## Metrics overwritten after resume
Process-local history could erase earlier learning-curve data.

**Lesson:** experiment history must outlive the process.

# Potential blog-post split

## Post 10 — The latest model isn't necessarily the best model
Validation-driven model selection, `latest.pt`, `best.pt`, and best-score persistence.

## Post 11 — What actually needs to survive when ML training resumes?
Model state, optimizer state, epoch, validation history, schema evolution.

## Post 12 — Stop editing magic numbers in your training script
`TrainingConfig`, immutability, reproducibility, and better config tests.

## Post 13 — Your terminal is not an experiment tracker
`EpochMetrics`, JSON history, resumable learning curves.

## Post 14 — An experiment should own its files
Experiment identity, scoped artifacts, and preventing stale-checkpoint contamination.

# Narrative angles

Across the project:

```text
Build Log 01:
Can the model learn?

Build Log 02:
Does it generalize?

Build Log 03:
Can I trust my experiments enough to tell whether the next model is better?
```

Another framing:

> Training a model is temporary. An experiment should survive the process that trained it.

Or:

> Once validation showed that the baseline was overfitting, the next problem wasn't immediately building a better neural network. It was making sure I could prove that the next neural network was actually better.

# Current experiment architecture

```text
TrainingConfig
      ↓
experiment identity
      ↓
artifacts/<experiment>/
      │
      ├── config.json
      ├── metrics.json
      ├── latest.pt
      └── best.pt
```

Training:

```text
PGN
 ↓
game-level split
 ↓
examples
 ↓
DataLoaders
 ↓
model
 ↓
loss
 ↓
backprop
 ↓
optimizer
 ↓
latest checkpoint
```

Evaluation:

```text
train loader ─────→ metrics
validation loader → metrics
                     ↓
                EpochMetrics
                     ↓
                metrics.json

validation improves?
        ↓
      best.pt
```

Resume:

```text
new process
    ↓
load latest.pt
    ├── model
    ├── optimizer
    ├── epoch
    └── best score

load metrics.json
    ↓
continue experiment
```

# What comes next

The infrastructure is now strong enough to return to the actual model.

The baseline established:

```text
training performance improves strongly
validation performance plateaus around 8–9% move accuracy
```

Future model changes can now be compared against that baseline with:

```text
same dataset split
same seed
known configuration
persisted metrics
best validation checkpoint
```

Likely future directions include:

```text
- richer board-state representation
- explicit side to move
- castling rights
- en passant state
- legal-move masking
- spatially aware board modeling
- stronger move decoding
- eventually attention / Transformer-style models
```

The important difference is that the project can now answer:

> Did this change actually improve unseen-game validation?

# Closing state

At the beginning of ChessSLM, success meant:

```text
loss went down
```

Then success became:

```text
validation improved
```

Now success means something stricter:

```text
I can reproduce the experiment,
resume it,
preserve its history,
identify its best model,
and compare it with the next experiment.
```

That is a much stronger foundation for the next stage of model development.
