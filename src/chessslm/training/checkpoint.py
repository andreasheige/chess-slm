from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_val_move_acc: float,
) -> None:
    """Save the current training state to disk.

    In addition to model and optimizer state, the checkpoint stores the
    best validation move accuracy seen so far.

    This allows a resumed training run to keep comparing against the same
    validation record instead of resetting "best" when the process restarts.
    """

    # Create the destination directory if it does not already exist.
    # This lets callers save to paths such as:
    # checkpoints/baseline.pt
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_move_acc": best_val_move_acc,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """Restore model and optimizer state from a checkpoint.

    Older checkpoints may not contain every field introduced in later
    versions of the training pipeline. Missing validation history falls
    back to 0.0 so older checkpoints can still be loaded.
    """

    # map_location controls which device the saved tensors are loaded onto.
    #
    # This matters because a checkpoint saved on one device should still
    # be loadable somewhere else, for example:
    #
    # MPS -> CPU
    # CUDA -> CPU
    # CPU -> MPS
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint.get("best_val_move_acc", 0.0)
