import chess

EMPTY = 0

PIECE_TO_ID = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}

ID_TO_PIECE = {value: key for key, value in PIECE_TO_ID.items()}


def encode_board(board: chess.Board) -> list[int]:
    encoded = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            encoded.append(EMPTY)
            continue

        piece_id = PIECE_TO_ID[(piece.color, piece.piece_type)]
        encoded.append(piece_id)

    return encoded


def decode_board(encoded: list[int]) -> chess.Board:
    if len(encoded) != 64:
        raise ValueError("Encoded board must contain exactly 64 squares")

    board = chess.Board.empty()

    for square, piece_id in enumerate(encoded):
        if piece_id == EMPTY:
            continue

        color, piece_type = ID_TO_PIECE[piece_id]
        board.set_piece_at(square, chess.Piece(piece_type, color))

    return board
