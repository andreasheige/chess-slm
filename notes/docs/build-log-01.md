# ChessSLM — Build Log 01
## From empty repository to a neural network that learns its first chess move

**Date:** 2026-10-03  
**Status:** First learning milestone reached  
**Purpose:** Raw material for future blog posts / technical write-ups

## Why this project exists

ChessSLM is primarily a learning vehicle for moving from senior software engineering into deeper ML/DL engineering. The goal is to implement and understand the important parts of the pipeline before hiding them behind high-level frameworks.

```text
raw chess data → representation → dataset → tensors → neural network
→ loss → backpropagation → optimization → evaluation → larger models
```

Core rule: **understand the mechanism before introducing the abstraction.**

## 1. Project bootstrap

The repository was set up with Python 3.12, `uv`, PyTorch, NumPy, `python-chess`, pytest, Ruff, Git/GitHub Actions, and Apple MPS support.

The package evolved toward:

```text
src/chessslm/
├── __init__.py
├── cli.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── examples.py
├── models/
│   ├── __init__.py
│   └── baseline.py
└── representation/
    ├── __init__.py
    ├── board.py
    └── move.py
```

Tests live in `tests/`; exploratory code lives under `scripts/`.

## 2. Guardrails

Codex is used as a pair programmer rather than an autopilot.

Key rules:

- One conceptual change at a time.
- Stop at phase boundaries.
- Explain tensor shapes.
- Do not introduce unexplained dependencies.
- Understand bugs before fixing them.
- Do not scale before validating the current implementation.
- Never suppress warnings just to make output clean.
- Preserve useful failures and experiments.

## 3. First chess data: PGN

The first dataset was deliberately tiny: one game, the Opera Game.

Before ML abstractions, the raw formats were understood:

- **PGN** — stores the game.
- **SAN** — human-oriented move notation, e.g. `Nf3`.
- **UCI** — explicit source/destination, e.g. `g1f3`.
- **FEN** — serialized chess position.

Using `python-chess`, the game was replayed one move at a time:

```text
position_t → move_t
```

Examples:

```text
initial position      → e2e4
position after e2e4   → e7e5
position after e7e5   → g1f3
```

One game therefore produces many supervised examples.

## 4. Board representation v0

The first representation contains only the 64 squares:

```text
0  = empty
1  = white pawn
2  = white knight
3  = white bishop
4  = white rook
5  = white queen
6  = white king
7  = black pawn
8  = black knight
9  = black bishop
10 = black rook
11 = black queen
12 = black king
```

The IDs are **categorical IDs**, not numerical values.

Encoding and decoding were implemented and round-trip tested:

```text
chess.Board → 64 IDs → piece placement
```

The v0 representation intentionally omits side-to-move, castling rights, en passant, and move counters.

**Lesson:** a model cannot recover information that its representation never provides.

## 5. Move representation v0

Instead of one giant vocabulary of complete moves, the first move representation is structured:

```text
from_square
to_square
promotion
```

Squares map naturally to `0..63`.

Example:

```text
g1f3 → from=6, to=21
```

This prepares three model outputs:

```text
from_square → 64 classes
to_square   → 64 classes
promotion   → 5 classes
```

Promotion classes:

```text
0 = no promotion
1 = knight
2 = bishop
3 = rook
4 = queen
```

## 6. Representation failure: promotion

The first move representation stored only `(from_square, to_square)`.

A deliberate round-trip test for:

```text
e7e8q
```

failed because it decoded as:

```text
e7e8
```

The missing promotion information was the root cause. The representation was expanded to include promotion and the test returned to green.

**Lesson:** representation design puts a hard ceiling on what a model can learn or express.

## 7. Useful engineering failures

Several small failures were worth preserving:

### Missing function looked like an import problem

`ImportError: cannot import name 'encode_board'`

Python had loaded the module correctly; the function simply had not been added.

### File in the wrong folder

`[Errno 2] No such file or directory`

Filesystem placement problem, not a Python/ML problem.

### Duplicate test names

Two tests had the same Python function name. Python replaced the first definition, so pytest stayed green. Ruff caught the redefinition.

**Lesson:** green tests do not guarantee every intended test was collected.

### Missing NumPy warning

PyTorch warned that NumPy could not initialize. NumPy was added as a real dependency rather than suppressing the warning.

## 8. TrainingExample and Dataset

One observation was formalized as:

```python
@dataclass(frozen=True)
class TrainingExample:
    board: list[int]
    from_square: int
    to_square: int
    promotion: int | None
```

A PGN game became:

```text
list[TrainingExample]
```

A PyTorch `ChessDataset` then converted examples into tensors:

```text
board        shape [64]   dtype torch.long
from_square  shape []     dtype torch.long
to_square    shape []     dtype torch.long
promotion    shape []     dtype torch.long
```

`torch.long` is used because these values are categorical/index targets.

## 9. DataLoader and batching

A `DataLoader` batched examples automatically:

```text
single board: [64]

four boards:
[4, 64]
 ↑   ↑
 │   └── squares
 └────── batch
```

A real batch from the Opera Game produced:

```text
board shape:        [4, 64]
from_square shape:  [4]
to_square shape:    [4]
promotion shape:    [4]
```

The first four numerical moves corresponded to:

```text
12 → 28   e2 → e4
52 → 36   e7 → e5
6  → 21   g1 → f3
51 → 43   d7 → d6
```

Pipeline at this point:

```text
PGN → TrainingExample → ChessDataset → DataLoader → batch tensors
```

## 10. First embedding

The first learnable neural component:

```python
nn.Embedding(num_embeddings=13, embedding_dim=4)
```

There are 13 board tokens and four learned values per token:

```text
embedding.weight shape = [13, 4]
parameters = 13 × 4 = 52
```

A real chess batch changed shape:

```text
[4, 64]
   ↓ embedding
[4, 64, 4]
```

Interpretation:

```text
4 boards × 64 squares × 4 learned features
```

This was the transition from symbolic IDs to a learned numerical representation.

## 11. First neural network

A deliberately simple baseline was built:

```text
board IDs [B,64]
      ↓
Embedding
      ↓
[B,64,4]
      ↓
Flatten
      ↓
[B,256]
      ↓
Linear(256 → 128)
      ↓
ReLU
      ↓
shared representation [B,128]
      │
      ├── from head      → [B,64]
      ├── to head        → [B,64]
      └── promotion head → [B,5]
```

No Transformer or attention yet.

The purpose is to prove the learning pipeline before adding architectural complexity.

## 12. Logits and cross entropy

The network outputs raw class scores called **logits**.

The model deliberately does not apply softmax itself because `CrossEntropyLoss` expects raw logits.

The relationship was verified manually:

```text
logits
  ↓ softmax
probability(correct class)
  ↓ -log(...)
cross entropy loss
```

One experiment produced:

```text
correct-class probability ≈ 0.0100
manual loss              = 4.6034
CrossEntropyLoss          = 4.6034
```

For 64 roughly equally likely classes, random behavior is around:

```text
ln(64) ≈ 4.16
```

So losses around four before training were expected.

## 13. Backpropagation and gradients

Before:

```python
loss.backward()
```

embedding gradients were `None`.

After backward, gradient tensors existed.

With an all-zero board, only embedding row `0` received gradient because only token `0` was used.

With a real starting position, all token types in that position received gradient signal.

This made the relationship concrete:

```text
forward pass → loss → backward → gradients
```

## 14. Optimizer step

SGD was introduced:

```python
torch.optim.SGD(model.parameters(), lr=0.1)
```

Weights were cloned before and after `optimizer.step()` and visibly changed.

The complete mechanism now existed:

```text
forward
  ↓
loss
  ↓
backward
  ↓
gradients
  ↓
optimizer.step()
  ↓
updated parameters
```

## 15. First actual learning

A dedicated script, `scripts/model/overfit_single_example.py`, asked one question:

> Can the model memorize one chess example?

The example:

```text
initial position → e2e4
```

Repeated runs showed the source-square loss falling from roughly four to near zero, with the prediction converging to square `12` (`e2`).

A random seed was then added:

```python
torch.manual_seed(42)
```

before model creation so the experiment became reproducible.

## 16. Training the complete move

Finally, all three heads were trained together.

For `e2e4`:

```text
from_square = 12
to_square   = 28
promotion   = 0
```

Three losses were summed:

```python
loss = from_loss + to_loss + promotion_loss
```

Observed run:

```text
step=  0 loss=10.7586 move=54→23 promotion=2
step= 10 loss=0.0006 move=12→28 promotion=0
step= 20 loss=0.0005 move=12→28 promotion=0
step= 50 loss=0.0003 move=12→28 promotion=0
step=100 loss=0.0002 move=12→28 promotion=0
```

The randomly initialized model began with nonsense and quickly memorized:

```text
12 → 28, promotion 0
```

which is exactly:

```text
e2 → e4
```

# Milestone: first learning

The complete supervised-learning loop now works:

```text
PGN
 ↓
position + move
 ↓
numeric representation
 ↓
Dataset
 ↓
DataLoader
 ↓
tensors
 ↓
Embedding
 ↓
neural network
 ↓
logits
 ↓
cross-entropy loss
 ↓
backpropagation
 ↓
gradients
 ↓
optimizer
 ↓
updated weights
 ↓
lower loss
```

The model **cannot play chess yet**.

What has been proven is more fundamental:

> The pipeline can learn.

## Screenshot / artifact candidates

Useful material to preserve for later writing:

1. Raw Opera Game PGN.
2. First FEN/SAN/UCI examples.
3. 64-value board encoding.
4. Failing promotion round-trip test.
5. Clean pytest run.
6. DataLoader `[4,64]` batch output.
7. Embedding `[4,64,4]` output.
8. Gradients before/after `backward()`.
9. Weights before/after `optimizer.step()`.
10. Single-example training log at steps 0, 10 and 100.
11. Baseline architecture diagram.

## Possible future blog split

### Post 1 — Building an ML project without starting with the model
Tooling, guardrails, PGN, representations, representation failures.

### Post 2 — From chess games to PyTorch tensors
TrainingExample, Dataset, DataLoader, shape, dtype, batching.

### Post 3 — My first neural network from first principles
Embeddings, parameters, baseline architecture, logits, cross entropy.

### Post 4 — Watching a neural network learn its first chess move
Autograd, gradients, optimizer, random seeds, deliberate overfitting.

## Narrative angle

A useful framing:

> I already know how to build and operate software systems. I wanted to understand what changes when the system itself learns its behavior from data.

The bridge from software engineering to ML can be described as:

```text
explicit rules        → learned parameters
data structures       → tensors
function output        → logits
error measurement     → loss
manual correction     → gradients
code/config change    → optimizer update
```

## Current state

Completed:

```text
✓ repository/toolchain
✓ tests/linting/formatting
✓ PGN parsing
✓ board representation
✓ move/promotion representation
✓ TrainingExample
✓ PyTorch Dataset
✓ DataLoader / batching
✓ tensors
✓ embeddings
✓ baseline neural network
✓ logits
✓ cross entropy
✓ backpropagation
✓ gradients
✓ optimizer
✓ reproducible initialization
✓ intentional overfitting of one complete move
```

Still intentionally missing:

```text
- training on multiple examples
- train/validation split
- validation loop
- meaningful accuracy metrics
- legal-move masking
- larger datasets
- full chess-state metadata
- checkpoints
- MPS training integration
- attention / Transformer
- Stockfish evaluation
```

# Next step

Move from:

```text
one position → one move → repeat until memorized
```

to:

```text
Opera Game examples
→ ChessDataset
→ DataLoader
→ batches
→ training loop
→ combined loss
```

The next question is:

> Can the same network intentionally overfit a small collection of real chess positions rather than one hard-coded move?
 