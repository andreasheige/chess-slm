import torch
from torch import nn
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import TrainingExample
from chessslm.models.baseline import ChessBaseline
from chessslm.training.train import evaluate, train_epoch, training_step

TEST_DEVICE = torch.device("cpu")


def test_training_step_returns_scalar_loss() -> None:
    model = ChessBaseline()
    loss_fn = nn.CrossEntropyLoss()

    batch = {
        "board": torch.zeros(
            (2, 64),
            dtype=torch.long,
        ),
        "from_square": torch.tensor(
            [12, 52],
            dtype=torch.long,
        ),
        "to_square": torch.tensor(
            [28, 36],
            dtype=torch.long,
        ),
        "promotion": torch.tensor(
            [0, 0],
            dtype=torch.long,
        ),
        "side_to_move": torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    }

    loss = training_step(
        model,
        batch,
        loss_fn,
    )

    assert loss.ndim == 0
    assert loss.item() > 0


def test_train_epoch_returns_mean_loss() -> None:
    examples = [
        TrainingExample(
            board=[0] * 64,
            from_square=12,
            to_square=28,
            promotion=None,
            side_to_move=0,
            en_passant_square=64,
        ),
        TrainingExample(
            board=[0] * 64,
            from_square=52,
            to_square=36,
            promotion=None,
            side_to_move=0,
            en_passant_square=64,
        ),
    ]

    dataset = ChessDataset(examples)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = ChessBaseline()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
    )
    loss_fn = nn.CrossEntropyLoss()

    mean_loss = train_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        TEST_DEVICE,
    )

    assert mean_loss > 0


def test_evaluate_returns_accuracies() -> None:
    examples = [
        TrainingExample(
            board=[0] * 64,
            from_square=12,
            to_square=28,
            promotion=None,
            side_to_move=0,
            en_passant_square=64,
        ),
        TrainingExample(
            board=[0] * 64,
            from_square=52,
            to_square=36,
            promotion=None,
            side_to_move=0,
            en_passant_square=64,
        ),
    ]

    dataset = ChessDataset(examples)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    model = ChessBaseline()

    from_accuracy, to_accuracy, move_accuracy = evaluate(
        model,
        loader,
        TEST_DEVICE,
    )

    assert 0.0 <= from_accuracy <= 1.0
    assert 0.0 <= to_accuracy <= 1.0
    assert 0.0 <= move_accuracy <= 1.0
