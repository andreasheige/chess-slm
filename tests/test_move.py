import chess

from chessslm.representation.move import (
    decode_move,
    decode_promotion,
    encode_move,
    encode_promotion,
)


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


def test_encode_no_promotion() -> None:
    assert encode_promotion(None) == 0


def test_encode_queen_promotion() -> None:
    assert encode_promotion(chess.QUEEN) == 4


def test_promotion_id_round_trip() -> None:
    for promotion in [
        None,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ]:
        encoded = encode_promotion(promotion)
        decoded = decode_promotion(encoded)

        assert decoded == promotion
