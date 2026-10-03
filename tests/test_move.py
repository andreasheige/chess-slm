import chess

from chessslm.representation.move import decode_move, encode_move


def test_encode_move() -> None:
    move = chess.Move.from_uci("g1f3")

    from_square, to_square, promotion = encode_move(move)

    assert from_square == 6
    assert to_square == 21
    assert promotion is None


def test_move_round_trip() -> None:
    original = chess.Move.from_uci("g1f3")

    from_square, to_square, promotion = encode_move(original)
    decoded = decode_move(from_square, to_square, promotion)

    assert decoded == original
    assert decoded.uci() == "g1f3"


def test_promotion_round_trip() -> None:
    original = chess.Move.from_uci("e7e8q")

    from_square, to_square, promotion = encode_move(original)
    decoded = decode_move(from_square, to_square, promotion)

    assert decoded == original
