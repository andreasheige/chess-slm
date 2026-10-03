import json
from dataclasses import asdict
from pathlib import Path

from chessslm.training.config import TrainingConfig
from chessslm.training.metrics import EpochMetrics


def save_experiment_artifacts(
    directory: Path,
    config: TrainingConfig,
    history: list[EpochMetrics],
) -> None:
    """Save experiment configuration and metric history as JSON files.

    The goal is to preserve enough information to understand and compare
    a training run later without relying on terminal output.

    Two files are written:

    - config.json  -> experiment configuration
    - metrics.json -> evaluation history across epochs
    """

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_data = asdict(config)

    # Path objects are not directly JSON serializable, so convert them
    # to strings before writing the configuration.
    config_data = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config_data.items()
    }

    metrics_data = [asdict(metrics) for metrics in history]

    with (directory / "config.json").open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config_data,
            config_file,
            indent=2,
        )

    with (directory / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics_data,
            metrics_file,
            indent=2,
        )


def load_metrics_history(
    directory: Path,
) -> list[EpochMetrics]:
    """Load metric history from a previous part of an experiment.

    Training may be stopped and resumed across multiple processes.
    Loading the existing history lets new evaluation points be appended
    without losing metrics collected before the restart.

    If no metrics file exists, the experiment is treated as a fresh run.
    """

    metrics_path = directory / "metrics.json"

    if not metrics_path.exists():
        return []

    with metrics_path.open(
        encoding="utf-8",
    ) as metrics_file:
        metrics_data = json.load(metrics_file)

    return [EpochMetrics(**metrics) for metrics in metrics_data]
