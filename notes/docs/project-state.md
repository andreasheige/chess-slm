# ChessSLM — Project State / Chat Handoff

**Updated:** 2026-10-04  
**Purpose:** Authoritative handoff for continuing ChessSLM in a new ChatGPT conversation.

## Project goal

ChessSLM is Andreas's first ML/DL project. The goal is to learn ML/DL while building a sound supervised chess move-prediction system with PyTorch and python-chess.

```text
chess position
→ neural network
→ from-square
→ to-square
→ promotion
```

Tooling: Python, PyTorch, python-chess, uv, pytest, Ruff, Git. Development machine supports Apple MPS.

## Teaching / collaboration style

This is important.

Andreas is experienced in software development, but this is his first ML/DL project. Keep the work pedagogical and incremental:

```text
1. Explain WHY the next change is needed.
2. Work on ONE file / concept at a time.
3. Show the complete relevant function/block when useful.
4. Run ONE focused test.
5. Get it green.
6. Only then move to the next layer.
```

Do not jump across `test_baseline.py`, `training_step()`, `evaluate()`, and `run.py` in one response.

Short comments explaining ML/PyTorch concepts are welcome. Avoid comments that merely narrate trivial Python.

## Current data pipeline

```text
multi-game PGN
      ↓
load_games()
      ↓
list[Game]
      ↓
split_games()
   ↙             ↘
train games      validation games
   ↓                  ↓
examples             examples
   ↓                  ↓
ChessDataset         ChessDataset
   ↓                  ↓
DataLoader           DataLoader
   ↓
device transfer
   ↓
ChessBaseline
```

Train/validation splitting is done at complete-game level to avoid leakage between related positions from the same game.

Current dataset:

```text
198 games
158 training games
40 validation games
13,951 training examples
2,983 validation examples
```

PGN:

```text
data/raw/MacKenzie.pgn
```

Raw PGN data is ignored by Git.

## Current move/model design

Targets:

```text
from_square
to_square
promotion
```

Full move accuracy requires all three to be correct.

Destination prediction is conditional on the source square:

```text
P(from | position)
P(to | position, from)
```

The correct `from_square` is supplied during training/evaluation (teacher forcing).

The active baseline representation currently contains:

```text
board piece IDs
→ piece embeddings
→ flattened board

side_to_move
→ learned embedding

flattened board + side embedding
→ hidden layer

hidden → from_head
hidden + from_square embedding → to_head
hidden → promotion_head
```

`side_to_move`:

```text
0 = White
1 = Black
```

## Baseline experiment history

### baseline_v1 — board-only

Best unseen-game full-move validation accuracy was approximately:

```text
~8.7–8.9%
```

Clear overfitting: training accuracy continued rising while validation plateaued early.

### baseline_v2_side_to_move — strongest baseline

Change:

```text
board + side_to_move
```

Best result:

```text
best val_move = 10.09% @ epoch 30
```

Selected values:

```text
epoch 10:
val_from = 21.89%
val_to   = 23.57%
val_move = 9.72%

epoch 30:
val_move = 10.09%  ← best

epoch 100:
train_move = 65.83%
val_move   = 8.85%
```

Conclusion:

> Explicit side-to-move information produced a measurable generalization improvement.

This is the current strongest baseline.

### baseline_v3_castling_rights — neutral/negative experiment

Change:

```text
board + side_to_move + 4 castling-right flags
```

Best result:

```text
best val_move = 9.99% @ epoch 70
```

Compared with v2:

```text
10.09%
```

Conclusion:

> No clear measurable gain from castling rights under the current dataset/model/training setup.

The castling implementation was committed, measured, and then reverted so the next isolated feature experiment could build from v2.

Recent history included:

```text
545e094 chore: track baseline experiment metrics
6d85c70 feat: add castling rights to position context
4846b3b feat: add side-to-move board context
3db4774 docs: build log 03
fba7108 feat: add reproducible experiment artifacts
```

`git revert 6d85c70` was performed. After the revert:

```text
35 tests passed
```

## Experiment infrastructure

Each experiment owns:

```text
artifacts/<experiment_name>/
├── config.json
├── metrics.json
├── latest.pt
└── best.pt
```

Semantics:

```text
config.json  → intended experiment configuration
metrics.json → learning/evaluation history
latest.pt    → resume state
best.pt      → best validation checkpoint
```

`.pt` files are ignored by Git.

Meaningful experiment `config.json` and `metrics.json` files are version-controlled. Temporary smoke-test experiment folders should not be kept.

### TrainingConfig

Defined in:

```text
src/chessslm/training/config.py
```

It is a frozen dataclass.

Conceptually:

```text
TrainingConfig → WHAT experiment to run
run.py         → HOW to execute it
```

Important config values:

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

Checkpoint paths are derived from `artifact_dir / experiment_name`, not separately stored in config.

Typical controlled comparison:

```text
batch_size = 8
epochs = 100
learning_rate = 0.001
validation_fraction = 0.2
seed = 42
log_every = 10
```

### Checkpoints

Checkpoint state includes:

```text
model_state_dict
optimizer_state_dict
epoch
best_val_move_acc
```

The loader is backward-compatible with old checkpoints lacking `best_val_move_acc`, falling back to `0.0`.

Important rule:

> A checkpoint belongs to the experiment/data split/configuration that created it.

### Metrics

`EpochMetrics` stores:

```text
epoch
loss
train_from_acc
train_to_acc
train_move_acc
val_from_acc
val_to_acc
val_move_acc
```

Metrics history survives resume. A run split across multiple Python processes retains one continuous `metrics.json`.

## Quality gates

Use:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

There is also a pre-commit quality gate.

Preferred experiment discipline:

```text
implementation
→ focused tests
→ full tests
→ lint/format
→ commit implementation
→ run experiment
→ inspect/version small JSON results
```

## Build logs

Existing historical logs:

```text
ChessSLM_BUILD_LOG_01.md
ChessSLM_BUILD_LOG_02.md
ChessSLM_BUILD_LOG_03.md
```

Conceptually:

```text
01: Can the model learn?
02: Does it generalize?
03: Can experiments be trusted, resumed and compared?
```

These are historical/blog raw material. This PROJECT_STATE file is the live handoff.

# CURRENT PHASE — Phase 20: en passant state

Active code is based on the v2 side-to-move baseline.

The next isolated feature experiment is:

```text
v2 + en passant state
```

Goal:

> Determine whether explicit en-passant state improves unseen-game validation while everything else stays constant.

## Representation decision

Encode en-passant state as one integer:

```text
0–63 = en-passant square
64   = no en-passant square
```

The location matters, so a boolean alone would lose information. Sentinel `64` also avoids carrying `None` into tensors/embeddings.

## Phase 20 completed so far

### TrainingExample / example creation

`TrainingExample` now has:

```python
en_passant_square: int
```

`create_training_example()` derives it from the board before `board.push(move)`:

```python
en_passant_square = board.ep_square if board.ep_square is not None else 64
```

### Behavior test

A test using a sequence such as:

```text
1. e4 d5
```

verifies:

```text
before e4:
en_passant_square = 64

after e4 / before Black's move:
en_passant_square = chess.E3
```

The examples tests were green.

### ChessDataset

`ChessDataset.__getitem__()` has been updated/planned to expose:

```python
"en_passant_square": torch.tensor(
    example.en_passant_square,
    dtype=torch.long,
)
```

# EXACT CURRENT STOPPING POINT

Most recent focused command:

```bash
uv run pytest tests/test_dataset.py
```

Result:

```text
3 failed
```

All failures have the same expected cause:

```text
TrainingExample.__init__()
missing required argument:
'en_passant_square'
```

Affected tests:

```text
test_dataset_length
test_dataset_returns_tensors_by_index
test_dataloader_batches_examples
```

## NEXT ACTION — do this first in the new chat

Update every manually constructed `TrainingExample(...)` in:

```text
tests/test_dataset.py
```

with:

```python
en_passant_square = 64
```

These tests are not testing actual en-passant behavior, so sentinel `64` is appropriate.

In `test_dataset_returns_tensors_by_index()` also verify:

```python
assert item["en_passant_square"].shape == torch.Size([])
assert item["en_passant_square"].dtype == torch.long
assert item["en_passant_square"].item() == 64
```

In `test_dataloader_batches_examples()` verify:

```python
assert batch["en_passant_square"].shape == torch.Size([4])
```

Then run ONLY:

```bash
uv run pytest tests/test_dataset.py
```

Expected:

```text
3 passed
```

STOP THERE before modifying the model.

## Intended Phase 20 sequence after that

```text
20.1 TrainingExample / creation       ✓
20.2 ep_square behavior test          ✓
20.3 ChessDataset tensorization       ← CURRENT
20.4 model representation
20.5 training_step
20.6 evaluate
20.7 full test/lint/format gate
20.8 commit implementation
20.9 run isolated experiment
20.10 compare with v2
```

A likely model design later is:

```text
en_passant_square
→ embedding with 65 entries
→ concatenate with board + side-to-move representation
```

Do NOT jump to this until the dataset tests are green and the next step is explained.

## Experiment naming

Historical experiment name `baseline_v3_castling_rights` already exists.

For the en-passant experiment, use a distinct name, likely:

```text
baseline_v4_en_passant
```

even though active code was reverted to v2 first.

Keep all other settings identical to v2.

Score to beat:

```text
baseline_v2_side_to_move
best val_move = 10.09%
```

# Key lessons to preserve

```text
- Evaluate on held-out games, not only training data.
- Split complete games before creating positions.
- Do not reuse stale checkpoints across experiments.
- latest.pt and best.pt have different responsibilities.
- Resume includes model, optimizer, epoch, best score and metrics history.
- Keep experiment parameters in TrainingConfig.
- Test contracts, not arbitrary temporary values such as epochs == 40.
- Scope checkpoints to experiment directories.
- Persist metrics; terminal output is not an experiment tracker.
- Negative experiments are useful.
- Build the next isolated experiment from the strongest baseline, not automatically the latest feature.
- Work one file/concept at a time while learning ML/DL.
```

# Suggested first message in the next chat

> We are continuing my ChessSLM project. This is my first ML/DL project. Read the attached `ChessSLM_PROJECT_STATE.md` first and continue exactly from the CURRENT STOPPING POINT. Keep the teaching style slow and explicit: one file/concept at a time, explain why, make the change, run one focused test, get it green, then continue. Do not jump ahead across multiple layers.
