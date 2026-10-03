import torch
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import TrainingExample


def test_dataset_length() -> None:
    examples = [
        TrainingExample(
            board=[0] * 64,
            from_square=6,
            to_square=21,
            promotion=None,
        ),
        TrainingExample(
            board=[0] * 64,
            from_square=12,
            to_square=28,
            promotion=None,
        ),
    ]

    dataset = ChessDataset(examples)

    assert len(dataset) == 2


def test_dataset_returns_tensors_by_index() -> None:
    example = TrainingExample(
        board=[0] * 64,
        from_square=6,
        to_square=21,
        promotion=None,
    )

    dataset = ChessDataset([example])
    item = dataset[0]

    assert item["board"].shape == torch.Size([64])
    assert item["board"].dtype == torch.long

    assert item["from_square"].shape == torch.Size([])
    assert item["from_square"].dtype == torch.long
    assert item["from_square"].item() == 6

    assert item["to_square"].shape == torch.Size([])
    assert item["to_square"].dtype == torch.long
    assert item["to_square"].item() == 21

    assert item["promotion"].shape == torch.Size([])
    assert item["promotion"].dtype == torch.long
    assert item["promotion"].item() == 0


def test_dataloader_batches_examples() -> None:
    examples = [
        TrainingExample(
            board=[0] * 64,
            from_square=6,
            to_square=21,
            promotion=None,
        ),
        TrainingExample(
            board=[1] * 64,
            from_square=12,
            to_square=28,
            promotion=None,
        ),
        TrainingExample(
            board=[2] * 64,
            from_square=1,
            to_square=18,
            promotion=None,
        ),
        TrainingExample(
            board=[3] * 64,
            from_square=57,
            to_square=42,
            promotion=None,
        ),
    ]

    dataset = ChessDataset(examples)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))

    assert batch["board"].shape == torch.Size([4, 64])
    assert batch["from_square"].shape == torch.Size([4])
    assert batch["to_square"].shape == torch.Size([4])
    assert batch["promotion"].shape == torch.Size([4])
