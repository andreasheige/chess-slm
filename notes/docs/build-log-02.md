# ChessSLM — Build Log 02
## From a one-move experiment to a real training pipeline — and the first generalization wall

**Date:** 2026-10-03  
**Status:** Multi-game training + unseen-game validation working  
**Previous log:** `ChessSLM_BUILD_LOG_01.md`  
**Purpose:** Raw material for future blog posts / technical write-ups

---

# Where Build Log 01 ended

Build Log 01 ended with a randomly initialized neural network learning one complete chess move. The pipeline could already go from PGN data to tensors, embeddings, logits, cross-entropy loss, gradients, optimizer updates, and lower loss.

For the initial position, the model learned:

```text
e2 → e4
promotion = none
```

That proved the pipeline could learn. Build Log 02 is about turning that experiment into the beginning of an actual ML engineering system — and then discovering that learning the training data is not the same thing as generalizing.

# 1. From scripts to reusable training code

Training behavior was moved from exploratory scripts into:

```text
src/chessslm/training/
├── __init__.py
├── train.py
└── run.py
```

The first reusable unit was `training_step()`: given a model and one batch, compute the combined loss for `from_square`, `to_square`, and `promotion`.

The important separation was:

```text
training_step()
→ compute loss

train_epoch()
→ zero gradients
→ compute loss
→ backward
→ optimizer step
→ repeat for every batch
```

Evaluation became a separate operation using `model.eval()` and `torch.no_grad()`. It reports `from_accuracy`, `to_accuracy`, and full `move_accuracy`.

A full move is correct only when source, destination, and promotion are all correct.

# 2. A real training entrypoint

A real orchestration entrypoint was added at:

```text
src/chessslm/training/run.py
```

Its job is now to load data, create datasets/loaders, construct the model, loss and optimizer, run epochs, evaluate, log metrics, and checkpoint state.

The old `overfit_small_dataset.py` experiment was removed once its responsibilities had moved into reusable project code. Git history remains the archive for that experiment.

# 3. Conditional destination prediction

The original baseline predicted:

```text
P(from | board)
P(to   | board)
```

On the small Opera Game overfit experiment it reached roughly:

```text
from_acc = 100.00%
to_acc   = 72.73%
move_acc = 72.73%
```

Nine errors remained, all with the correct source square but the wrong destination. That suggested the destination head needed explicit source-square information.

The model was changed to:

```text
P(from | board)
P(to   | board, from)
```

A source-square embedding was introduced and teacher forcing supplies the correct `from_square` during training.

Conceptually:

```text
board → shared representation ─────────→ from_head
                  │
correct from → embedding
                  │
                  └─ concatenate ──────→ to_head

shared representation ─────────────────→ promotion_head
```

On the same small experiment, full move accuracy improved from about `72.73%` to `81.82%`, and remaining errors dropped from 9 to 6.

This was a useful controlled experiment: same data and training setup, one structural change, measurable improvement.

# 4. Device support

A central device selector was introduced with the priority:

```text
CUDA → MPS → CPU
```

On the development Mac:

```text
get_device() → mps
```

The important lesson was that both model parameters and batch tensors must live on the same device.

```text
DataLoader → CPU tensors
              ↓
move_batch_to_device()
              ↓
           MPS tensors
              ↓
         model on MPS
```

Tests deliberately remain CPU-based so the suite behaves consistently locally and in CI.

The full training pipeline was verified end-to-end on Apple MPS.

# 5. Checkpoints: training stopped being disposable

Checkpoint save/load was added.

A checkpoint stores:

```text
model_state_dict
optimizer_state_dict
epoch
```

Model state preserves learned parameters. Optimizer state matters because optimizers such as Adam keep internal history used by future updates.

The first checkpoint:

```text
checkpoints/latest.pt
```

was approximately 603 KB.

## Checkpoint failures worth keeping

The first save was accidentally placed before the training loop. That would have saved random weights while claiming epoch 100.

A second bug passed a string where the checkpoint helper expected a `Path`:

```text
AttributeError: 'str' object has no attribute 'parent'
```

Both were simple software bugs, but useful reminders that ML systems still depend on ordinary type and control-flow correctness.

# 6. Resume training

Checkpoint loading was connected to the training entrypoint:

```text
process starts
→ checkpoint exists?
→ restore model
→ restore optimizer
→ restore epoch
→ continue from next epoch
```

A successful run loaded epoch 100, continued through epochs 110 and 120, and saved a new checkpoint.

Training could now survive process termination instead of always starting from random weights.

# 7. Quality gates moved closer to development

The project already used:

```text
ruff check
ruff format --check
pytest
```

Ruff was configured for editor-save formatting/fixes, and a pre-commit hook was added so commits run lint, format verification, and tests.

The resulting feedback layers:

```text
save
→ fast Ruff feedback

commit
→ Ruff + tests

push / PR
→ CI verifies again
```

The first pre-commit installation collided with an existing Git `core.hooksPath` configuration. The configuration source had to be removed before pre-commit could manage the repository hooks.

# 8. The validation boundary

Until this point, accuracy was measured on the same examples used for training.

That answers:

> Can the model fit this data?

It does not answer:

> Can the model generalize?

A deterministic train/validation split was introduced.

With the single Opera Game:

```text
26 training examples
7 validation examples
```

Training accuracy approached 100%, while validation move accuracy remained 0%.

With only seven validation examples, one correct prediction would change accuracy by about 14.29 percentage points, so the result was conceptually useful but statistically tiny.

Still, it was the project's first direct demonstration of memorization versus generalization.

# 9. Checkpoints belong to experiments

The first validation run tried to resume a checkpoint previously trained on all 33 Opera Game positions.

That invalidated the intended validation boundary: those seven validation examples were no longer unseen.

Important rule:

> A checkpoint belongs to the data split and experiment configuration that created it.

Changing the validation boundary can make an old checkpoint unusable for clean evaluation.

The old checkpoint was discarded and validation training restarted from fresh weights.

# 10. Multi-game PGN loading

The project then moved beyond a single game.

A PGN can contain multiple games, so the data layer gained:

```python
load_games(...)
```

Conceptually:

```text
PGN
 ↓
Game
Game
Game
...
```

A test with two small games verified the multi-game loader.

# 11. Split by game, not by position

Randomly splitting positions from the same game risks leakage because neighboring chess positions are highly related.

Instead, complete games are split first:

```text
Game A → train
Game B → train
Game C → validation
```

Only after the split are games converted into supervised position/move examples.

The pipeline became:

```text
multi-game PGN
      ↓
load_games()
      ↓
list[Game]
      ↓
split_games()
   ↙            ↘
train games     validation games
   ↓                  ↓
create examples   create examples
   ↓                  ↓
train dataset     validation dataset
```

This makes validation substantially more meaningful: validation positions come from games that never contribute optimizer updates.

# 12. Data-layer refactor failure

While adding `split_games()`, `split_examples()` was accidentally changed to accept `chess.pgn.Game` instead of `TrainingExample`.

The two related concepts had been mixed together.

The final module deliberately preserves both contracts:

```text
split_examples()
→ position/example-level split

split_games()
→ complete-game-level split
```

Small tests around data boundaries caught the mistake quickly.

# 13. Multiple games → one dataset

A helper `create_examples_from_games()` was added.

```text
Game 1 → examples ─┐
Game 2 → examples ─┼→ combined examples
Game 3 → examples ─┘
```

Crucially, this happens after train and validation games have already been separated.

# 14. First real multi-game dataset

The first meaningful multi-game run used:

```text
198 games total
158 training games
40 validation games
```

After conversion:

```text
13,951 training examples
2,983 validation examples
```

With `batch_size=8`:

```text
1,744 training batches per epoch
```

ChessSLM had moved from one game and a few dozen examples to roughly 17,000 supervised positions across 198 games.

# 15. Stale checkpoint: zero training iterations

The first multi-game run loaded an old checkpoint at epoch 100 while the configured target was also epoch 100.

That produced:

```text
start_epoch = 101
range(101, 101)
```

which contains zero iterations.

The process loaded and saved a checkpoint without training.

More importantly, the checkpoint belonged to the previous dataset anyway and should not have been reused.

This reinforced the idea that checkpoint state is experiment-specific.

# 16. First meaningful generalization curve

A fresh multi-game run showed:

```text
epoch   train_move   val_move
0          5.33%       4.89%
10        16.47%       8.62%
20        26.85%       8.65%
50        43.75%       8.15%
80        52.56%       8.78%
100       54.81%       8.51%
```

Training performance kept improving.

Validation improved early, then flattened.

```text
epoch 0 → ~10
training ↑
validation ↑

epoch ~10 → 100
training ↑↑↑
validation ─ / ↓
```

This was ChessSLM's first clear generalization gap.

# 17. Better metrics showed where the gap exists

Logging was expanded to include:

```text
train_from
train_to
train_move

val_from
val_to
val_move
```

At epoch 0:

```text
train_from = 13.25%
train_to   = 14.61%
train_move =  5.33%

val_from   = 11.46%
val_to     = 13.64%
val_move   =  4.89%
```

At epoch 10:

```text
train_from = 35.09%
train_to   = 40.80%
train_move = 16.51%

val_from   = 16.90%
val_to     = 22.59%
val_move   =  8.62%
```

At epoch 100:

```text
train_from = 74.75%
train_to   = 73.46%
train_move = 57.04%

val_from   = 15.72%
val_to     = 19.64%
val_move   =  8.65%
```

This showed that the generalization problem is broader than the destination head.

For source-square prediction:

```text
train: ~35% → ~75%
validation: ~17% → ~16%
```

For destination prediction:

```text
train: ~41% → ~73%
validation: ~23% → ~20%
```

The model increasingly fits the training games while unseen-game performance stops improving.

# 18. Why more epochs are not the obvious answer

Without validation, `train_move_acc ≈ 57%` might suggest simply training longer.

Validation changes the interpretation.

After roughly ten epochs:

```text
validation move accuracy ≈ 8–9%
```

Additional optimization improves training fit but not generalization.

```text
more epochs
→ better training fit
→ almost no validation improvement
```

The baseline has therefore done its job: it provides a measurable reference point and exposes its limitations.

# 19. Current baseline limitations

The board representation remains deliberately primitive:

```text
64 categorical piece IDs
```

It does not explicitly represent:

```text
side to move
castling rights
en passant state
move counters
```

The baseline model is also simple:

```text
piece IDs
→ embeddings
→ flatten board
→ one hidden linear layer
→ prediction heads
```

It has no explicit mechanism for:

```text
spatial board relationships
rank/file structure
piece movement geometry
attack/defense relationships
legal move structure
board symmetries
attention between squares
```

This is not a project failure. It is exactly what a baseline is supposed to reveal.

# Milestone reached — multi-game baseline with unseen-game validation

The system now supports:

```text
✓ reusable training_step
✓ reusable train_epoch
✓ reusable evaluation
✓ documented training entrypoint
✓ CPU / MPS / CUDA-aware device selection
✓ explicit batch device movement
✓ checkpoint save/load/resume
✓ pre-commit quality gate
✓ deterministic splitting
✓ multi-game PGN loading
✓ game-level train/validation split
✓ multi-game example creation
✓ unseen-game validation
✓ per-head train/validation metrics
✓ measurable generalization gap
```

Current dataset:

```text
198 games

158 training games
40 validation games

13,951 training positions
2,983 validation positions
```

Current baseline after 100 epochs:

```text
TRAIN
from accuracy ≈ 74.75%
to accuracy   ≈ 73.46%
move accuracy ≈ 57.04%

VALIDATION
from accuracy ≈ 15.72%
to accuracy   ≈ 19.64%
move accuracy ≈ 8.65%
```

# Failures worth preserving

- Checkpoint saved before training.
- String passed where `Path` was expected.
- Resume run with no epochs left.
- `saved_epoch` referenced when no checkpoint existed.
- Evaluation block accidentally placed outside the training loop.
- Old checkpoint contaminating a new validation experiment.
- Position-level split becoming inappropriate for multi-game data.
- `split_examples()` and `split_games()` temporarily mixed together.
- Old checkpoint silently producing a zero-iteration multi-game run.

These are useful because they show that ML engineering is not separate from software engineering. Experimental validity depends on ordinary code correctness.

# Visuals worth preserving

1. Transition from experiment scripts to `src/chessslm/training/`.
2. Conditional destination architecture.
3. First successful `Device: mps` run.
4. `checkpoints/latest.pt` (~603 KB).
5. Resume log from epoch 100 to 120.
6. First train/validation split.
7. 100% train vs 0% validation on the tiny single-game split.
8. Multi-game counts: 198 / 158 / 40.
9. Position counts: 13,951 / 2,983.
10. Training-vs-validation move accuracy curve.
11. From/to train-vs-validation metrics.

# Suggested chart data

```text
epoch | train_move | val_move
0     | 5.33       | 4.89
10    | 16.51      | 8.62
20    | 27.09      | 8.31
30    | 34.08      | 8.51
40    | 39.70      | 8.45
50    | 45.60      | 8.55
60    | 48.48      | 8.25
70    | 51.73      | 8.85
80    | 53.71      | 8.35
90    | 56.32      | 8.31
100   | 57.04      | 8.65
```

The story is immediately visible:

```text
training keeps improving
validation stops improving
```

# Possible blog-post split

## Post 5 — Turning an ML experiment into a training system
`training_step`, `train_epoch`, evaluation, reusable architecture, removing obsolete experiments.

## Post 6 — Devices, checkpoints and resumable training
CPU/MPS/CUDA, tensor devices, state dicts, optimizer state, save/load/resume.

## Post 7 — The validation set changed the answer
Training accuracy vs validation accuracy, memorization, checkpoint contamination.

## Post 8 — Scaling from one chess game to 198
Multi-game PGN, game-level split, leakage, ~17k positions, first meaningful validation curve.

## Post 9 — My model is learning. That's not the same as getting better.
Generalization gap, from/to metrics, overfitting, why more epochs are not automatically useful.

# Narrative angle

A useful continuation from Build Log 01:

> The first milestone was getting a neural network to learn. The second was discovering that learning is the easy part. The harder question is whether it learns anything that survives outside the data it trained on.

Or even more simply:

```text
Build Log 01:
"Look — the loss goes down."

Build Log 02:
"Okay, but does that mean anything?"
```

# Current architecture

```text
board IDs [B,64]
      ↓
piece embedding
      ↓
[B,64,E]
      ↓
flatten
      ↓
shared hidden representation
      │
      ├────────────────────→ from head
      │
source-square embedding
      │
      └── concatenate ─────→ to head
      │
      └────────────────────→ promotion head
```

Training uses teacher forcing for destination prediction:

```text
correct from_square
→ source-square embedding
→ destination prediction
```

# Current engineering pipeline

```text
multi-game PGN
      ↓
load_games()
      ↓
split complete games
   ↙               ↘
train games       validation games
   ↓                    ↓
create examples      create examples
   ↓                    ↓
Dataset              Dataset
   ↓                    ↓
DataLoader           DataLoader
   ↓
device transfer
   ↓
model
   ↓
loss
   ↓
backpropagation
   ↓
optimizer
   ↓
checkpoint

periodically:
model
   ↓
train evaluation
validation evaluation
   ↓
metrics
```

# What comes next

The infrastructure is now strong enough that future model changes can be measured against a real baseline.

The next phase should distinguish:

```text
latest.pt
→ latest training state
→ used to resume training

best.pt
→ best validation result
→ used for model selection
```

The current learning curve already shows why the latest model is not necessarily the model that generalizes best.

After that, representation and architecture changes can be introduced one controlled experiment at a time:

```text
- richer board state
- side-to-move
- castling / en passant state
- legal-move masking
- spatially aware representations
- stronger move decoding
- attention / Transformer architecture
- larger datasets
```

The next step is not to jump directly to the most sophisticated architecture.

The next step is to preserve the baseline properly so future changes can prove whether they actually improve generalization.

# Closing state

ChessSLM started as:

```text
Can I make a neural network learn e2e4?
```

It has now become:

```text
Can I build a reproducible training system,
train across complete chess games,
hold out unseen games,
resume experiments,
and measure whether the model generalizes?
```

The answer is now yes.

And the baseline has delivered its first genuinely useful negative result:

> The model learns the training games far better than it learns transferable chess structure.

That is exactly the kind of result a baseline is supposed to reveal.
