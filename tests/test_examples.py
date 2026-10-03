import chess
import chess.pgn

from chessslm.data.examples import (
    TrainingExample,
    create_examples_from_game,
    create_training_example,
)


def test_training_example() -> None:
    example = TrainingExample(
        board=[0] * 64,
        from_square=6,
        to_square=21,
        promotion=None,
    )

    assert len(example.board) == 64
    assert example.from_square == 6
    assert example.to_square == 21
    assert example.promotion is None


def test_create_training_example() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    example = create_training_example(board, move)

    assert len(example.board) == 64
    assert example.from_square == chess.E2
    assert example.to_square == chess.E4
    assert example.promotion is None


def test_create_examples_from_game() -> None:
    with open("data/raw/sample.pgn") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    examples = create_examples_from_game(game)

    assert len(examples) == 33

    first = examples[0]
    assert first.from_square == chess.E2
    assert first.to_square == chess.E4

    third = examples[2]
    assert third.from_square == chess.G1
    assert third.to_square == chess.F3
