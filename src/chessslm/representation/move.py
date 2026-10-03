import chess


def encode_move(move: chess.Move) -> tuple[int, int, int | None]:
    return move.from_square, move.to_square, move.promotion


def decode_move(from_square: int, to_square: int, promotion: int | None) -> chess.Move:
    return chess.Move(from_square, to_square, promotion=promotion)
