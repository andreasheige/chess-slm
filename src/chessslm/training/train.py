from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from chessslm.models.baseline import ChessBaseline


def training_step(
    model: ChessBaseline,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Compute the combined loss for one training batch."""

    from_logits, to_logits, promotion_logits = model(
        batch["board"],
        batch["from_square"],
    )

    from_loss = loss_fn(
        from_logits,
        batch["from_square"],
    )

    to_loss = loss_fn(
        to_logits,
        batch["to_square"],
    )

    promotion_loss = loss_fn(
        promotion_logits,
        batch["promotion"],
    )

    return from_loss + to_loss + promotion_loss


def train_epoch(
    model: ChessBaseline,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: Optimizer,
    loss_fn: nn.CrossEntropyLoss,
) -> float:
    """Train the model for one complete pass over the dataset."""

    model.train()

    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        # Gradients accumulate in PyTorch, so clear the previous
        # batch before calculating new gradients.
        optimizer.zero_grad()

        loss = training_step(
            model,
            batch,
            loss_fn,
        )

        # Compute gradients for all trainable parameters.
        loss.backward()

        # Update the model using those gradients.
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count


def evaluate(
    model: ChessBaseline,
    loader: Iterable[dict[str, torch.Tensor]],
) -> tuple[float, float, float]:
    """Evaluate from-square, to-square, and full-move accuracy."""

    model.eval()

    from_correct = 0
    to_correct = 0
    move_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            from_logits, to_logits, promotion_logits = model(
                batch["board"],
                batch["from_square"],
            )

            from_prediction = torch.argmax(
                from_logits,
                dim=1,
            )
            to_prediction = torch.argmax(
                to_logits,
                dim=1,
            )
            promotion_prediction = torch.argmax(
                promotion_logits,
                dim=1,
            )

            from_match = from_prediction == batch["from_square"]
            to_match = to_prediction == batch["to_square"]
            promotion_match = promotion_prediction == batch["promotion"]

            from_correct += from_match.sum().item()
            to_correct += to_match.sum().item()

            move_match = from_match & to_match & promotion_match

            move_correct += move_match.sum().item()
            total += batch["board"].shape[0]

    return (
        from_correct / total,
        to_correct / total,
        move_correct / total,
    )
