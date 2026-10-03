from pathlib import Path

import chess.pgn
import torch
from torch import nn
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import create_examples_from_game
from chessslm.data.split import split_examples
from chessslm.device import get_device
from chessslm.models.baseline import ChessBaseline
from chessslm.training.checkpoint import load_checkpoint, save_checkpoint
from chessslm.training.train import evaluate, train_epoch

# For now we deliberately train on one very small PGN file.
#
# The goal at this stage is not to build a strong chess model.
# We want to verify that the complete training pipeline works
# before introducing larger datasets and more complex models.
PGN_PATH = "data/raw/sample.pgn"
CHECKPOINT_PATH = Path("checkpoints/latest.pt")


def main() -> None:
    # Select the best available compute device.
    #
    # On the current Mac this should resolve to MPS.
    # On other machines it can fall back to CPU or use CUDA.
    device = get_device()

    print(f"Device: {device}")

    # Neural network parameters are randomly initialized.
    #
    # Setting a seed makes that initialization reproducible,
    # which makes repeated experiments easier to compare.
    torch.manual_seed(42)

    # Read one chess game from disk.
    #
    # python-chess parses the PGN and gives us a Game object that
    # can be replayed move by move.
    with open(PGN_PATH) as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    # Convert the game into supervised learning examples:
    #
    # board position -> correct move
    #
    # One chess game therefore produces many training examples.
    examples = create_examples_from_game(game)

    # Split the examples into:
    #
    # training data:
    #   used for gradient updates
    #
    # validation data:
    #   never used for optimizer updates; only used to measure
    #   how well the model performs on unseen examples
    training_examples, validation_examples = split_examples(
        examples,
        validation_fraction=0.2,
        seed=42,
    )

    # ChessDataset converts our domain-level TrainingExample objects
    # into tensors that PyTorch can work with.
    training_dataset = ChessDataset(training_examples)
    validation_dataset = ChessDataset(validation_examples)

    # Training data is shuffled so the model does not always see
    # examples in the same order.
    training_loader = DataLoader(
        training_dataset,
        batch_size=8,
        shuffle=True,
    )

    # Validation data is not shuffled because we are only evaluating it.
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
    )

    # Create a fresh neural network and move its parameters
    # to the selected device.
    model = ChessBaseline().to(device)

    # CrossEntropyLoss measures how wrong each classification head is.
    #
    # Our model predicts:
    # - from_square
    # - to_square
    # - promotion
    #
    # training_step() combines the three losses.
    loss_fn = nn.CrossEntropyLoss()

    # Adam uses gradients from backpropagation to update the model.
    #
    # lr is the learning rate: roughly how aggressively parameters
    # are changed during each optimizer step.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # Start from epoch 0 unless an existing checkpoint is restored.
    start_epoch = 0

    # If a checkpoint exists, restore the previous training state.
    #
    # This restores:
    # - learned model parameters
    # - optimizer state
    # - the epoch where training stopped
    #
    # Training then continues from the following epoch.
    if CHECKPOINT_PATH.exists():
        saved_epoch = load_checkpoint(
            CHECKPOINT_PATH,
            model,
            optimizer,
            device,
        )

        start_epoch = saved_epoch + 1

        print(f"Checkpoint loaded: {CHECKPOINT_PATH} (epoch {saved_epoch})")

    # One epoch means one complete pass through the training dataset.
    epochs = 100

    print(f"Training examples: {len(training_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(f"Training batches per epoch: {len(training_loader)}")

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        # train_epoch performs:
        #
        # batch
        #   -> forward pass
        #   -> loss
        #   -> backward pass
        #   -> gradients
        #   -> optimizer step
        #
        # for every batch in the training DataLoader.
        mean_loss = train_epoch(
            model,
            training_loader,
            optimizer,
            loss_fn,
            device,
        )

        # Evaluate every ten epochs.
        #
        # Training accuracy tells us how well the model fits
        # examples it is allowed to learn from.
        #
        # Validation accuracy measures performance on examples
        # that never contribute gradients or optimizer updates.
        if epoch % 10 == 0:
            (
                _train_from_acc,
                _train_to_acc,
                train_move_acc,
            ) = evaluate(
                model,
                training_loader,
                device,
            )

            (
                _val_from_acc,
                _val_to_acc,
                val_move_acc,
            ) = evaluate(
                model,
                validation_loader,
                device,
            )

            print(
                f"epoch={epoch:3d} "
                f"loss={mean_loss:.4f} "
                f"train_move_acc={train_move_acc:.2%} "
                f"val_move_acc={val_move_acc:.2%}"
            )

    # Save the latest training state after the run has completed.
    #
    # The checkpoint contains:
    # - model parameters
    # - optimizer state
    # - final epoch
    save_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer,
        epoch=epochs,
    )

    print(f"Checkpoint saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
