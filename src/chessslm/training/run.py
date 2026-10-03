from pathlib import Path

import chess.pgn
import torch
from torch import nn
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import create_examples_from_game
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
    device = get_device()

    print(f"Device: {device}")
    # Neural network parameters are randomly initialized.
    #
    # Setting a seed makes that initialization reproducible,
    # which means repeated experiments start from the same weights
    # and become much easier to compare.
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

    # ChessDataset converts our domain-level TrainingExample objects
    # into tensors that PyTorch can work with.
    dataset = ChessDataset(examples)

    # DataLoader is responsible for batching and shuffling.
    #
    # batch_size=8 means the model sees eight positions at once.
    #
    # shuffle=True changes the example order between epochs.
    # This is typical during training and avoids always presenting
    # positions in the exact order they occurred in the chess game.
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    # Create a fresh neural network.
    #
    # At this point all learnable parameters contain random values.
    model = ChessBaseline().to(device)

    # CrossEntropyLoss measures how wrong each classification head is.
    #
    # Our model predicts:
    #
    # - from_square
    # - to_square
    # - promotion
    #
    # train_epoch() combines the three losses.
    loss_fn = nn.CrossEntropyLoss()

    # The optimizer uses gradients produced by backpropagation
    # to update the model's learnable parameters.
    #
    # Adam adapts the size of parameter updates during training.
    # lr is the learning rate: roughly how aggressively we update.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    start_epoch = 0

    # If a checkpoint already exists, restore the previous training state.
    #
    # This restores:
    # - learned model parameters
    # - optimizer state
    # - the epoch where training stopped
    #
    # We then continue from the following epoch instead of starting over.
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
    epochs = 120

    print(f"Training examples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    for epoch in range(start_epoch, epochs + 1):
        # train_epoch performs:
        #
        # batch
        #   -> forward pass
        #   -> loss
        #   -> backward pass
        #   -> gradients
        #   -> optimizer step
        #
        # for every batch in the DataLoader.
        mean_loss = train_epoch(model, loader, optimizer, loss_fn, device)

        # Evaluating every epoch would work for this tiny experiment,
        # but normally evaluation has a cost.
        #
        # Logging every ten epochs also keeps the output readable.
        if epoch % 10 == 0:
            (
                from_accuracy,
                to_accuracy,
                move_accuracy,
            ) = evaluate(model, loader, device)

            # from_acc:
            #   Did we predict the correct source square?
            #
            # to_acc:
            #   Did we predict the correct destination square?
            #
            # move_acc:
            #   Were from_square, to_square AND promotion all correct?
            print(
                f"epoch={epoch:3d} "
                f"loss={mean_loss:.4f} "
                f"from_acc={from_accuracy:.2%} "
                f"to_acc={to_accuracy:.2%} "
                f"move_acc={move_accuracy:.2%}"
            )
    # Save AFTER training has completed.
    save_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer,
        epoch=epochs,
    )
    print(f"Checkpoint saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
