import io

import chess
import chess.pgn

from chessslm.data.examples import (
    TrainingExample,
    create_examples_from_game,
    create_examples_from_games,
    create_training_example,
)


def test_training_example() -> None:
    example = TrainingExample(
        board=[0] * 64,
        from_square=6,
        to_square=21,
        promotion=None,
        side_to_move=0,
        white_kingside_castling=1,
        white_queenside_castling=1,
        black_kingside_castling=1,
        black_queenside_castling=1,
    )

    assert len(example.board) == 64
    assert example.from_square == 6
    assert example.to_square == 21
    assert example.promotion is None
    assert example.side_to_move == 0
    assert example.white_kingside_castling == 1
    assert example.white_queenside_castling == 1
    assert example.black_kingside_castling == 1
    assert example.black_queenside_castling == 1


def test_create_training_example() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    example = create_training_example(board, move)

    assert len(example.board) == 64
    assert example.from_square == chess.E2
    assert example.to_square == chess.E4
    assert example.promotion is None
    assert example.side_to_move == 0


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


def test_create_examples_from_games_combines_games() -> None:
    first_game = chess.pgn.read_game(io.StringIO("1. e4 e5 *"))
    second_game = chess.pgn.read_game(io.StringIO("1. d4 d5 *"))

    examples = create_examples_from_games([first_game, second_game])

    assert len(examples) == 4


def test_castling_rights_change_after_king_moves() -> None:
    game = chess.pgn.read_game(
        io.StringIO(
            """
[Result "*"]

1. e4 e5
2. Ke2 *
""".strip()
        )
    )

    examples = create_examples_from_game(game)

    # Before White's first move, both White castling rights exist.
    assert examples[0].white_kingside_castling == 1
    assert examples[0].white_queenside_castling == 1

    # Before White plays Ke2, the king is still on e1 and castling
    # rights still exist.
    assert examples[2].white_kingside_castling == 1
    assert examples[2].white_queenside_castling == 1


def test_castling_rights_are_removed_after_king_moves() -> None:
    game = chess.pgn.read_game(
        io.StringIO(
            """
[Result "*"]

1. e4 e5
2. Ke2 Nc6 *
""".strip()
        )
    )

    examples = create_examples_from_game(game)

    # Before White moves the king, both White castling rights exist.
    assert examples[2].white_kingside_castling == 1
    assert examples[2].white_queenside_castling == 1

    # The next example is created after Ke2 has been played.
    # White has permanently lost both castling rights.
    assert examples[3].white_kingside_castling == 0
    assert examples[3].white_queenside_castling == 0
