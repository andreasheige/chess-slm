from pathlib import Path

import torch

from chessslm.models.baseline import ChessBaseline
from chessslm.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def test_save_checkpoint_creates_file(
    tmp_path: Path,
) -> None:
    model = ChessBaseline()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=10,
        best_val_move_acc=0.25,
    )

    assert checkpoint_path.exists()


def test_load_checkpoint_restores_training_state(
    tmp_path: Path,
) -> None:
    original_model = ChessBaseline()

    original_optimizer = torch.optim.Adam(
        original_model.parameters(),
        lr=0.001,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        original_model,
        original_optimizer,
        epoch=42,
        best_val_move_acc=0.375,
    )

    # Create completely new objects.
    #
    # Their model weights are randomly initialized and therefore
    # initially different from the saved model.
    restored_model = ChessBaseline()

    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=0.001,
    )

    (restored_epoch, restored_best_val_move_acc) = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
    )

    assert restored_epoch == 42
    assert restored_best_val_move_acc == 0.375

    # Verify that every learned parameter was restored exactly.
    for original_parameter, restored_parameter in zip(
        original_model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.equal(
            original_parameter,
            restored_parameter,
        )
