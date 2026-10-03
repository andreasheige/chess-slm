import torch
from torch import nn
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import create_examples_from_games
from chessslm.data.pgn import load_games
from chessslm.data.split import split_games
from chessslm.device import get_device
from chessslm.models.baseline import ChessBaseline
from chessslm.training.artifacts import (
    load_metrics_history,
    save_experiment_artifacts,
)
from chessslm.training.checkpoint import load_checkpoint, save_checkpoint
from chessslm.training.config import TrainingConfig
from chessslm.training.metrics import EpochMetrics
from chessslm.training.train import evaluate, train_epoch


def main() -> None:
    # Select the best available compute device.
    #
    # On the current Mac this should resolve to MPS.
    # On other machines it can fall back to CPU or use CUDA.
    device = get_device()

    # TrainingConfig is a frozen dataclass that holds all the parameters
    # for a single training run.
    #
    # Keeping experiment parameters in one object makes training runs
    # easier to reproduce and compare.
    config = TrainingConfig()

    experiment_dir = config.artifact_dir / config.experiment_name

    latest_checkpoint_path = experiment_dir / "latest.pt"

    best_checkpoint_path = experiment_dir / "best.pt"

    # Neural network parameters are randomly initialized.
    #
    # Setting a seed makes that initialization reproducible,
    # which makes repeated experiments easier to compare.
    torch.manual_seed(42)

    # Load every game from the PGN file.
    #
    # We keep games intact at this stage because train/validation splitting
    # must happen before individual positions are extracted.
    games = load_games(config.pgn_path)

    # Split complete games rather than individual positions.
    #
    # This prevents positions from the same game appearing in both the
    # training and validation sets, which would leak information across
    # the evaluation boundary.
    training_games, validation_games = split_games(
        games, validation_fraction=config.validation_fraction, seed=config.seed
    )

    # Only after the game-level split do we turn each side into individual
    # supervised position -> move examples.
    training_examples = create_examples_from_games(training_games)

    validation_examples = create_examples_from_games(validation_games)

    # ChessDataset converts our domain-level TrainingExample objects
    # into tensors that PyTorch can work with.
    training_dataset = ChessDataset(training_examples)
    validation_dataset = ChessDataset(validation_examples)

    # Training data is shuffled so the model does not always see
    # examples in the same order.
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Validation data is not shuffled because we are only evaluating it.
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
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
        lr=config.learning_rate,
    )

    # Start from epoch 0 unless an existing checkpoint is restored.
    start_epoch = 0

    # Track the best validation result seen during this training run.
    #
    # Unlike latest.pt, best.pt is only replaced when validation
    # performance actually improves.
    best_val_move_acc = 0.0

    # If a checkpoint exists, restore the previous training state.
    #
    # This restores:
    # - learned model parameters
    # - optimizer state
    # - the epoch where training stopped
    #
    # Training then continues from the following epoch.
    if latest_checkpoint_path.exists():
        (saved_epoch, best_val_move_acc) = load_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            device,
        )

        start_epoch = saved_epoch + 1

        print(
            f"Checkpoint loaded: {latest_checkpoint_path} "
            f"(epoch {saved_epoch}, "
            f"best_val_move_acc={best_val_move_acc:.2%})"
        )

    print(f"Training examples: {len(training_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(f"Training batches per epoch: {len(training_loader)}")
    print(f"Games: {len(games)}")
    print(f"Training games: {len(training_games)}")
    print(f"Validation games: {len(validation_games)}")
    print(f"Device: {device}")
    print(f"Seed: {config.seed}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epoch target: {config.epochs}")

    # Restore metrics from earlier parts of this experiment.
    #
    # This keeps the learning curve continuous when training is resumed
    # from a checkpoint in a new process.
    history = load_metrics_history(experiment_dir)

    for epoch in range(
        start_epoch,
        config.epochs + 1,
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
        if epoch % config.log_every == 0:
            (
                train_from_acc,
                train_to_acc,
                train_move_acc,
            ) = evaluate(
                model,
                training_loader,
                device,
            )

            (
                val_from_acc,
                val_to_acc,
                val_move_acc,
            ) = evaluate(
                model,
                validation_loader,
                device,
            )

            metrics = EpochMetrics(
                epoch=epoch,
                loss=mean_loss,
                train_from_acc=train_from_acc,
                train_to_acc=train_to_acc,
                train_move_acc=train_move_acc,
                val_from_acc=val_from_acc,
                val_to_acc=val_to_acc,
                val_move_acc=val_move_acc,
            )

            history.append(metrics)

            print(
                f"epoch={epoch:3d} "
                f"loss={mean_loss:.4f} "
                f"train_from={train_from_acc:.2%} "
                f"train_to={train_to_acc:.2%} "
                f"train_move={train_move_acc:.2%} "
                f"val_from={val_from_acc:.2%} "
                f"val_to={val_to_acc:.2%} "
                f"val_move={val_move_acc:.2%}"
            )

            # Keep a separate checkpoint for the model that performs best on
            # validation games.
            #
            # Training accuracy is intentionally NOT used here. The purpose of
            # best.pt is to preserve the model that generalizes best to games
            # that were never used for optimizer updates.
            if val_move_acc > best_val_move_acc:
                best_val_move_acc = val_move_acc

                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    best_val_move_acc=best_val_move_acc,
                )

                print(
                    f"New best checkpoint: "
                    f"val_move_acc={best_val_move_acc:.2%} "
                    f"epoch={epoch}"
                )
    # Preserve the configuration and evaluation history for this experiment.
    #
    # This gives us structured data for later comparison and visualization
    # instead of relying only on terminal output.
    save_experiment_artifacts(
        experiment_dir,
        config,
        history,
    )

    print(f"Experiment artifacts saved: {experiment_dir}")

    # Save the latest training state after the run has completed.
    #
    # The checkpoint contains:
    # - model parameters
    # - optimizer state
    # - final epoch
    save_checkpoint(
        latest_checkpoint_path,
        model,
        optimizer,
        config.epochs,
        best_val_move_acc=best_val_move_acc,
    )

    print(f"Checkpoint saved: {latest_checkpoint_path}")


if __name__ == "__main__":
    main()
