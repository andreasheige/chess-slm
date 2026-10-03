import chess

from chessslm.representation.board import encode_board


def test_encode_starting_position_has_64_squares() -> None:
    board = chess.Board()

    encoded = encode_board(board)

    assert len(encoded) == 64


def test_encode_starting_position_first_rank() -> None:
    board = chess.Board()

    encoded = encode_board(board)

    assert encoded[:8] == [4, 2, 3, 5, 6, 3, 2, 4]


def test_encode_starting_position_second_rank() -> None:
    board = chess.Board()

    encoded = encode_board(board)

    assert encoded[8:16] == [1] * 8


def test_encode_starting_position_empty_middle() -> None:
    board = chess.Board()

    encoded = encode_board(board)

    assert encoded[16:48] == [0] * 32