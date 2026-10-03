from pathlib import Path

from chessslm.training.config import TrainingConfig


def test_training_config_has_valid_defaults() -> None:
    config = TrainingConfig()

    assert isinstance(config.pgn_path, Path)
    assert config.batch_size > 0
    assert config.epochs > 0
    assert config.learning_rate > 0.0
    assert 0.0 < config.validation_fraction < 1.0
    assert config.seed >= 0
    assert config.log_every > 0
    assert config.experiment_name

    assert isinstance(config.artifact_dir, Path)
