import chess.pgn

PGN_PATH = "data/raw/sample.pgn"


with open(PGN_PATH) as pgn_file:
    game = chess.pgn.read_game(pgn_file)

moves = [move.uci() for move in game.mainline_moves()]

print(f"Number of moves (plies): {len(moves)}")
print(f"Unique UCI moves: {len(set(moves))}")

print("\nMoves:")
for move in moves:
    print(move)
