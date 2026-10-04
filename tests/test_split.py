import chess.pgn

from chessslm.data.examples import TrainingExample
from chessslm.data.split import split_examples, split_games


def make_example(index: int) -> TrainingExample:
    return TrainingExample(
        board=[0] * 64,
        from_square=index,
        to_square=index + 1,
        promotion=None,
        side_to_move=0,
    )


def test_split_examples_separates_train_and_validation() -> None:
    examples = [make_example(index) for index in range(10)]

    training_examples, validation_examples = split_examples(
        examples,
        validation_fraction=0.2,
        seed=42,
    )

    assert len(training_examples) == 8
    assert len(validation_examples) == 2

    assert len(training_examples) + len(validation_examples) == len(examples)

    assert not {example.from_square for example in training_examples}.intersection(
        example.from_square for example in validation_examples
    )


def test_split_examples_is_reproducible() -> None:
    examples = [make_example(index) for index in range(10)]

    first_split = split_examples(
        examples,
        validation_fraction=0.2,
        seed=42,
    )

    second_split = split_examples(
        examples,
        validation_fraction=0.2,
        seed=42,
    )

    assert first_split == second_split


def make_game(name: str) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = name

    return game


def test_split_games_keeps_games_separate() -> None:
    games = [make_game(f"Game {index}") for index in range(10)]

    training_games, validation_games = split_games(
        games,
        validation_fraction=0.2,
        seed=42,
    )

    assert len(training_games) == 8
    assert len(validation_games) == 2

    training_names = {game.headers["Event"] for game in training_games}

    validation_names = {game.headers["Event"] for game in validation_games}

    assert training_names.isdisjoint(validation_names)
