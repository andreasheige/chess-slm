import chess.pgn
import torch
from torch import nn
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import create_examples_from_game
from chessslm.models.baseline import ChessBaseline

PGN_PATH = "data/raw/sample.pgn"

# Make the experiment reproducible.
# The model will start from the same random weights every run.
torch.manual_seed(42)


# Load one chess game and turn every position -> move pair
# into a supervised training example.
with open(PGN_PATH) as pgn_file:
    game = chess.pgn.read_game(pgn_file)

examples = create_examples_from_game(game)
dataset = ChessDataset(examples)


# Deliberately use the entire dataset as one batch.
#
# This is NOT how we expect to train a real model later.
# Right now we are running a diagnostic overfitting experiment:
# can this model memorize these 33 examples at all?
#
# Full batch + shuffle=False also removes batch-order noise,
# making the experiment easier to reason about.
loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=False,
)


model = ChessBaseline()
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
)


print(f"Training examples: {len(dataset)}")
print(f"Batches per epoch: {len(loader)}")


def evaluate(
    model: ChessBaseline,
    loader: DataLoader,
) -> tuple[float, float, float]:
    """Measure training accuracy without updating model parameters."""

    # Evaluation mode matters for modules such as dropout/batch norm.
    # Our current baseline does not use them yet, but establishing the
    # correct train/eval contract now makes the loop future-proof.
    model.eval()

    from_correct = 0
    to_correct = 0
    move_correct = 0
    total = 0

    # We only want predictions here, not gradients.
    # This avoids building an unnecessary autograd graph.
    with torch.no_grad():
        for batch in loader:
            from_logits, to_logits, promotion_logits = model(
                batch["board"],
                batch["from_square"],
            )

            from_prediction = torch.argmax(from_logits, dim=1)
            to_prediction = torch.argmax(to_logits, dim=1)
            promotion_prediction = torch.argmax(
                promotion_logits,
                dim=1,
            )

            from_match = from_prediction == batch["from_square"]
            to_match = to_prediction == batch["to_square"]
            promotion_match = promotion_prediction == batch["promotion"]

            from_correct += from_match.sum().item()
            to_correct += to_match.sum().item()

            # A complete move is only correct when all three
            # independently predicted components are correct.
            move_match = from_match & to_match & promotion_match
            move_correct += move_match.sum().item()

            total += batch["board"].shape[0]

    # Switch back because training continues after evaluation.
    model.train()

    return (
        from_correct / total,
        to_correct / total,
        move_correct / total,
    )


def inspect_errors(
    model: ChessBaseline,
    loader: DataLoader,
) -> None:
    """Print every training example the final model still gets wrong."""

    model.eval()

    print("\nPrediction errors:")

    error_count = 0

    with torch.no_grad():
        for batch in loader:
            from_logits, to_logits, promotion_logits = model(
                batch["board"],
                batch["from_square"],
            )

            from_predictions = torch.argmax(
                from_logits,
                dim=1,
            )
            to_predictions = torch.argmax(
                to_logits,
                dim=1,
            )
            promotion_predictions = torch.argmax(
                promotion_logits,
                dim=1,
            )

            for index in range(batch["board"].shape[0]):
                target_from = batch["from_square"][index].item()
                target_to = batch["to_square"][index].item()
                target_promotion = batch["promotion"][index].item()

                predicted_from = from_predictions[index].item()
                predicted_to = to_predictions[index].item()
                predicted_promotion = promotion_predictions[index].item()

                correct = (
                    predicted_from == target_from
                    and predicted_to == target_to
                    and predicted_promotion == target_promotion
                )

                if correct:
                    continue

                error_count += 1

                target_move = (
                    f"{chess.square_name(target_from)}{chess.square_name(target_to)}"
                )

                predicted_move = (
                    f"{chess.square_name(predicted_from)}"
                    f"{chess.square_name(predicted_to)}"
                )

                print(
                    f"{error_count:2d}. "
                    f"target={target_move} "
                    f"predicted={predicted_move} "
                    f"promotion="
                    f"{predicted_promotion}/{target_promotion}"
                )

    print(f"\nTotal errors: {error_count}")

    model.train()


# Deliberately train for many iterations.
#
# Because the DataLoader contains exactly one batch,
# one epoch == one optimizer update in this experiment.
for epoch in range(2501):
    total_loss = 0.0

    for batch in loader:
        # PyTorch accumulates gradients by default, so each update
        # must start by clearing gradients from the previous one.
        optimizer.zero_grad()

        # Forward pass.
        from_logits, to_logits, promotion_logits = model(
            batch["board"],
            batch["from_square"],
        )

        # Each component of the move is currently treated as its own
        # classification problem.
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

        # All three prediction heads contribute gradient signal
        # to the shared representation.
        loss = from_loss + to_loss + promotion_loss

        # Compute gradients and then update the model parameters.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    mean_loss = total_loss / len(loader)

    # Evaluation is relatively expensive and we do not need it
    # after every optimizer update. Every 250 epochs is enough
    # to see the learning trajectory clearly.
    if epoch % 250 == 0:
        (
            from_accuracy,
            to_accuracy,
            move_accuracy,
        ) = evaluate(
            model,
            loader,
        )

        print(
            f"epoch={epoch:4d} "
            f"loss={mean_loss:.4f} "
            f"from_acc={from_accuracy:.2%} "
            f"to_acc={to_accuracy:.2%} "
            f"move_acc={move_accuracy:.2%}"
        )


# Training accuracy tells us how many examples are wrong.
# This tells us WHICH examples are wrong, so the next model change
# can be driven by evidence rather than guessing.
inspect_errors(
    model,
    loader,
)
