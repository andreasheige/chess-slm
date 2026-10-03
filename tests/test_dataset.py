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


def test_dataset_returns_example_by_index() -> None:
    example = TrainingExample(
        board=[0] * 64,
        from_square=6,
        to_square=21,
        promotion=None,
    )

    dataset = ChessDataset([example])

    assert dataset[0] == example
