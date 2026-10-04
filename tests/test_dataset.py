import torch
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import TrainingExample


def test_dataset_length() -> None:
    """Dataset length should match the number of training examples."""

    examples = [
        TrainingExample(
            board=[0] * 64,
            from_square=6,
            to_square=21,
            promotion=None,
            side_to_move=0,
            white_kingside_castling=1,
            white_queenside_castling=1,
            black_kingside_castling=1,
            black_queenside_castling=1,
        ),
        TrainingExample(
            board=[0] * 64,
            from_square=12,
            to_square=28,
            promotion=None,
            side_to_move=1,
            white_kingside_castling=1,
            white_queenside_castling=0,
            black_kingside_castling=1,
            black_queenside_castling=0,
        ),
    ]

    dataset = ChessDataset(examples)

    assert len(dataset) == 2


def test_dataset_returns_tensors_by_index() -> None:
    """One dataset item should contain model-ready tensors.

    Scalar chess state such as side-to-move and castling rights is stored
    as scalar long tensors so PyTorch can batch and embed the values later.
    """

    example = TrainingExample(
        board=[0] * 64,
        from_square=6,
        to_square=21,
        promotion=None,
        side_to_move=0,
        white_kingside_castling=1,
        white_queenside_castling=0,
        black_kingside_castling=1,
        black_queenside_castling=0,
    )

    dataset = ChessDataset([example])
    item = dataset[0]

    # The board contains one encoded value for each of the 64 squares.
    assert item["board"].shape == torch.Size([64])
    assert item["board"].dtype == torch.long

    # Move targets are scalar class indices.
    assert item["from_square"].shape == torch.Size([])
    assert item["from_square"].dtype == torch.long
    assert item["from_square"].item() == 6

    assert item["to_square"].shape == torch.Size([])
    assert item["to_square"].dtype == torch.long
    assert item["to_square"].item() == 21

    # Promotion class 0 means no promotion.
    assert item["promotion"].shape == torch.Size([])
    assert item["promotion"].dtype == torch.long
    assert item["promotion"].item() == 0

    # 0 = White to move, 1 = Black to move.
    assert item["side_to_move"].shape == torch.Size([])
    assert item["side_to_move"].dtype == torch.long
    assert item["side_to_move"].item() == 0

    # Castling rights are binary state values:
    # 1 = the right still exists, 0 = it has been lost.
    assert item["white_kingside_castling"].shape == torch.Size([])
    assert item["white_kingside_castling"].dtype == torch.long
    assert item["white_kingside_castling"].item() == 1

    assert item["white_queenside_castling"].shape == torch.Size([])
    assert item["white_queenside_castling"].dtype == torch.long
    assert item["white_queenside_castling"].item() == 0

    assert item["black_kingside_castling"].shape == torch.Size([])
    assert item["black_kingside_castling"].dtype == torch.long
    assert item["black_kingside_castling"].item() == 1

    assert item["black_queenside_castling"].shape == torch.Size([])
    assert item["black_queenside_castling"].dtype == torch.long
    assert item["black_queenside_castling"].item() == 0


def test_dataloader_batches_examples() -> None:
    """DataLoader should combine individual examples into batch tensors."""

    examples = [
        TrainingExample(
            board=[0] * 64,
            from_square=6,
            to_square=21,
            promotion=None,
            side_to_move=0,
            white_kingside_castling=1,
            white_queenside_castling=1,
            black_kingside_castling=1,
            black_queenside_castling=1,
        ),
        TrainingExample(
            board=[1] * 64,
            from_square=12,
            to_square=28,
            promotion=None,
            side_to_move=1,
            white_kingside_castling=1,
            white_queenside_castling=0,
            black_kingside_castling=1,
            black_queenside_castling=0,
        ),
        TrainingExample(
            board=[2] * 64,
            from_square=1,
            to_square=18,
            promotion=None,
            side_to_move=0,
            white_kingside_castling=0,
            white_queenside_castling=0,
            black_kingside_castling=1,
            black_queenside_castling=1,
        ),
        TrainingExample(
            board=[3] * 64,
            from_square=57,
            to_square=42,
            promotion=None,
            side_to_move=1,
            white_kingside_castling=0,
            white_queenside_castling=0,
            black_kingside_castling=0,
            black_queenside_castling=0,
        ),
    ]

    dataset = ChessDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))

    # DataLoader adds the batch dimension.
    assert batch["board"].shape == torch.Size([4, 64])

    assert batch["from_square"].shape == torch.Size([4])
    assert batch["to_square"].shape == torch.Size([4])
    assert batch["promotion"].shape == torch.Size([4])

    # Scalar game-state values also become one value per example.
    assert batch["side_to_move"].shape == torch.Size([4])
    assert batch["white_kingside_castling"].shape == torch.Size([4])
    assert batch["white_queenside_castling"].shape == torch.Size([4])
    assert batch["black_kingside_castling"].shape == torch.Size([4])
    assert batch["black_queenside_castling"].shape == torch.Size([4])
