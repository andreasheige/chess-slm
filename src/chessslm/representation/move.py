import chess

PROMOTION_TO_ID = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


ID_TO_PROMOTION = {value: key for key, value in PROMOTION_TO_ID.items()}


def encode_move(move: chess.Move) -> tuple[int, int, int | None]:
    return move.from_square, move.to_square, move.promotion


def decode_move(from_square: int, to_square: int, promotion: int | None) -> chess.Move:
    return chess.Move(from_square, to_square, promotion=promotion)


def encode_promotion(promotion: int | None) -> int:
    return PROMOTION_TO_ID[promotion]


def decode_promotion(promotion_id: int) -> int | None:
    return ID_TO_PROMOTION[promotion_id]
