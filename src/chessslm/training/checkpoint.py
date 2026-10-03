from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
) -> None:
    """Save the current training state to disk.

    A checkpoint lets us stop training and later continue from the
    same state instead of starting again with random model weights.

    We save more than just the model parameters:

    - model_state_dict contains the learned model parameters.
    - optimizer_state_dict contains Adam/SGD's internal state.
    - epoch records where training stopped.

    Together these describe the important state needed to resume
    this training run.
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
) -> int:
    """Restore model and optimizer state from a saved checkpoint.

    Loading a checkpoint lets us continue a previous training run
    instead of creating a new model with randomly initialized weights.

    The model and optimizer objects are created by the caller first.
    This function then replaces their current state with the values
    stored in the checkpoint.

    The returned epoch tells the caller where the saved run stopped.
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

    return checkpoint["epoch"]
