import json
from pathlib import Path

from chessslm.training.artifacts import (
    load_metrics_history,
    save_experiment_artifacts,
)
from chessslm.training.config import TrainingConfig
from chessslm.training.metrics import EpochMetrics


def test_save_experiment_artifacts_creates_json_files(
    tmp_path: Path,
) -> None:
    config = TrainingConfig()

    history = [
        EpochMetrics(
            epoch=10,
            loss=1.23,
            train_from_acc=0.5,
            train_to_acc=0.6,
            train_move_acc=0.4,
            val_from_acc=0.3,
            val_to_acc=0.35,
            val_move_acc=0.2,
        )
    ]

    save_experiment_artifacts(
        tmp_path,
        config,
        history,
    )

    config_path = tmp_path / "config.json"
    metrics_path = tmp_path / "metrics.json"

    assert config_path.exists()
    assert metrics_path.exists()

    config_data = json.loads(config_path.read_text())

    metrics_data = json.loads(metrics_path.read_text())

    assert config_data["batch_size"] == 8
    assert config_data["learning_rate"] == 0.001

    assert metrics_data[0]["epoch"] == 10
    assert metrics_data[0]["val_move_acc"] == 0.2


def test_load_metrics_history_restores_saved_metrics(
    tmp_path: Path,
) -> None:
    config = TrainingConfig()

    original_history = [
        EpochMetrics(
            epoch=10,
            loss=1.23,
            train_from_acc=0.5,
            train_to_acc=0.6,
            train_move_acc=0.4,
            val_from_acc=0.3,
            val_to_acc=0.35,
            val_move_acc=0.2,
        )
    ]

    save_experiment_artifacts(
        tmp_path,
        config,
        original_history,
    )

    restored_history = load_metrics_history(tmp_path)

    assert restored_history == original_history


def test_load_metrics_history_returns_empty_for_new_experiment(
    tmp_path: Path,
) -> None:
    history = load_metrics_history(tmp_path)

    assert history == []
