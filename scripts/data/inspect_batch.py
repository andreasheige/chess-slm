import chess.pgn
from torch.utils.data import DataLoader

from chessslm.data.dataset import ChessDataset
from chessslm.data.examples import create_examples_from_game

PGN_PATH = "data/raw/sample.pgn"


with open(PGN_PATH) as pgn_file:
    game = chess.pgn.read_game(pgn_file)

examples = create_examples_from_game(game)
dataset = ChessDataset(examples)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

batch = next(iter(loader))

print("board shape:", batch["board"].shape)
print("from_square shape:", batch["from_square"].shape)
print("to_square shape:", batch["to_square"].shape)
print("promotion shape:", batch["promotion"].shape)

print("\nfrom_square:", batch["from_square"])
print("to_square:", batch["to_square"])
print("promotion:", batch["promotion"])
