import chess.pgn


PGN_PATH = "data/raw/sample.pgn"


with open(PGN_PATH) as pgn_file:
    game = chess.pgn.read_game(pgn_file)

board = game.board()

for ply, move in enumerate(game.mainline_moves(), start=1):
    fen = board.fen()
    san = board.san(move)
    uci = move.uci()

    print(f"\nTraining example {ply}")
    print(f"Input FEN: {fen}")
    print(f"Target SAN: {san}")
    print(f"Target UCI: {uci}")

    board.push(move)

    if ply == 3:
        break

print("Initial position:")
print(board)