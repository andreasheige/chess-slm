from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one training experiment.

    Keeping experiment parameters in one object makes training runs
    easier to reproduce and compare.

    These values describe how the experiment should run. They are kept
    separate from the model's learned parameters and optimizer state.
    """

    pgn_path: Path = Path("data/raw/MacKenzie.pgn")

    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.001

    validation_fraction: float = 0.2
    seed: int = 42

    log_every: int = 10

    experiment_name: str = "baseline_v3_castling_rights"
    artifact_dir: Path = Path("artifacts")
