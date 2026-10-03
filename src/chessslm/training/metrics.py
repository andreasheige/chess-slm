from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics captured at one evaluation point during training."""

    epoch: int
    loss: float

    train_from_acc: float
    train_to_acc: float
    train_move_acc: float

    val_from_acc: float
    val_to_acc: float
    val_move_acc: float
