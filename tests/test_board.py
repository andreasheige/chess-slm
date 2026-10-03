import chess

from chessslm.representation.board import decode_board, encode_board


def test_encode_starting_position_has_64_squares() -> None:
    board = chess.Board()

    encoded = encode_board(board)

    assert len(encoded) == 64


def test_board_round_trip() -> None:
    original = chess.Board()

    encoded = encode_board(original)
    decoded = decode_board(encoded)

    assert decoded.board_fen() == original.board_fen()


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


def test_board_round_trip_after_moves() -> None:
    original = chess.Board()

    original.push_uci("e2e4")
    original.push_uci("e7e5")
    original.push_uci("g1f3")

    encoded = encode_board(original)
    decoded = decode_board(encoded)

    assert decoded.board_fen() == original.board_fen()
