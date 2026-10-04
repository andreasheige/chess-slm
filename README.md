# ChessSLM

**Learning machine learning from first principles, through chess.**

A small, fully tested PyTorch pipeline that goes from raw PGN games to
tensors, a trained move-prediction model, and honest evaluation, built one
validated step at a time on the way to a transformer and an autoregressive
chess language model.

> **Status:** work in progress. No playing-strength claims are made; see
> [Limits of the current results](#limits-of-the-current-results).

## Why this project

I come from web development and spent the last years in DevOps, DevSecOps and
developer experience. ChessSLM is where I apply that engineering discipline
to ML: reproducible splits, pinned dependencies (`uv.lock`), pre-commit, CI,
unit tests, and written design decisions ([ADRs](notes/decisions/)) before
scaling anything up.

## Roadmap

- [x] PGN loading and position–move examples
- [x] Reproducible game-level train/validation split
- [x] Board and move encoding, PyTorch dataset and loaders
- [x] Baseline model (embeddings + linear) with three prediction heads
- [x] Training loop, accuracy metrics, checkpointing, CUDA/MPS/CPU selection
- [x] Tests, Ruff, pre-commit, GitHub Actions
- [ ] Full board state in the encoding (side to move, castling, en passant)
- [ ] Inference-time evaluation and legal-move masking
- [ ] Experiment tracking (dataset identity, split config, RNG state)
- [ ] Attention / Transformer model
- [ ] Autoregressive modeling
- [ ] Stockfish evaluation, distillation, self-play, serving, MLOps

Details in the [implementation plan](IMPLEMENTATION_PLAN.md).

## Results

| Model | Source acc. | Destination acc. | Full-move acc. | Data |
|---|---|---|---|---|
| Baseline (embeddings + linear) | _x %_ | _x %_ | _x %_ | _n games_ |

_Teacher-forced evaluation, piece placement only. Fill in with your latest run._

## Setup

Use Python 3.12 or newer and `uv`. The repository pins Python 3.12 in
`.python-version` and commits `uv.lock` for dependency reproducibility.

From the repository root:

```sh
uv python install
uv sync --locked --dev
```

Check that the package entry point works:

```sh
uv run chessslm
```

This currently prints a greeting. Training is run separately as a Python module.

## Inspect the data

A small PGN fixture is included at `data/raw/sample.pgn`. These scripts expose
the early steps of the pipeline:

```sh
uv run python scripts/data/inspect_pgn.py
uv run python scripts/data/inspect_moves.py
uv run python scripts/data/inspect_batch.py
```

They show board tokens, played moves, and batch tensor shapes. Run them from the
repository root so their relative data paths resolve correctly.

## Run training

The training runner currently reads `data/raw/MacKenzie.pgn`. This is a local
dataset and is **not included in Git**. Place a multi-game PGN at that path, or
change `PGN_PATH` in [run.py](src/chessslm/training/run.py) to your own file.
Use at least two games so both training and validation contain data; the
single-game sample fixture is intended for inspection rather than this runner.

```sh
uv run python -m chessslm.training.run
```

The runner splits complete games before extracting positions, reserving 20% of
games for validation with seed 42. It uses batches of 8, Adam with a learning
rate of `0.001`, and reports loss and accuracies every ten epochs. The current
epoch range includes epochs 0 through 100 for a fresh run. These settings are
defined directly in the runner rather than through command-line options.

Training saves to `checkpoints/latest.pt` and automatically loads that file on
subsequent runs. To start a fresh experiment after changing the model or data,
move the existing checkpoint aside. To continue training beyond its saved
epoch, increase the `epochs` value in the runner.

Raw datasets other than the sample fixture, processed data, and checkpoints are
ignored by Git.

## How the baseline works

Each board is represented by 64 integer tokens in square order `a1` through
`h8`: `0` means empty, and `1`–`12` identify piece and color combinations.
Move targets contain a source square, a destination square, and one of five
promotion classes: none, knight, bishop, rook, or queen.

For a batch of size `B`, the model uses this flow:

```text
board tokens [B, 64]
    → piece embeddings [B, 64, 4]
    → flatten [B, 256]
    → linear layer + ReLU [B, 128]
    → source-square logits [B, 64]
    → destination-square logits [B, 64], conditioned on a source-square embedding
    → promotion logits [B, 5]
```

Training sums cross-entropy losses for the three prediction heads. The
destination head receives the correct source square during training, a technique
called *teacher forcing*.

### Limits of the current results

- The input contains piece placement only. Side to move, castling rights,
  en passant state, and move counters are omitted, so distinct chess states can
  share the same encoding.
- Evaluation also supplies the correct source square to the destination head.
  Full-move accuracy requires all three predicted targets to match, but it does
  not measure inference where the destination is conditioned on the model's
  own source prediction.
- Predictions are not constrained to legal moves. Accuracy measures agreement
  with recorded moves, rather than move quality or playing strength.
- Keeping games separate prevents positions from the same game crossing the
  split. It does not guarantee that identical positions never occur in different
  games on both sides.
- Checkpoints do not record dataset identity, split configuration, or random
  number generator state, so they are not complete experiment records.

These limitations are part of the current baseline and should inform how its
metrics are interpreted.

## Development checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

To enable the local pre-commit hooks:

```sh
uv run pre-commit install
```

The hooks and CI run linting, formatting checks, and tests.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/chessslm/data/` | PGN loading, example construction, splits, and dataset |
| `src/chessslm/representation/` | Board and move encoding |
| `src/chessslm/models/` | Neural network baseline |
| `src/chessslm/training/` | Training, evaluation, and checkpoints |
| `scripts/` | Small inspection and learning scripts |
| `tests/` | Unit tests for the pipeline |
| `notes/` | Design decisions and build notes |
| `data/raw/` | Sample fixture and local PGN datasets |

Some exploratory scripts reflect earlier model interfaces and may need updating
as the baseline changes.

## Direction

The [implementation plan](IMPLEMENTATION_PLAN.md) describes the wider roadmap:
validate the small pipeline, deliberately overfit tiny data, establish a
reproducible baseline, then explore better representations, attention,
Transformers, and autoregressive modeling. Stockfish evaluation, distillation,
self-play, serving, and MLOps are later planned stages.

Progress is guided by understanding and validating each step before scaling.
The roadmap describes intended work, not a list of completed features.

See the [initial learning-objective decision](notes/decisions/ADR-0001-initial-learning-objective.md)
for the reasoning behind the first task and its simplified representation.
