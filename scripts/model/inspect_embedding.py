import chess.pgn
from torch import nn
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

embedding = nn.Embedding(
    num_embeddings=13,
    embedding_dim=4,
)

board = batch["board"]
embedded_board = embedding(board)

print("Board:")
print("shape:", board.shape)
print("dtype:", board.dtype)

print("\nEmbedded board:")
print("shape:", embedded_board.shape)
print("dtype:", embedded_board.dtype)

print("\nEmbedding weights:")
print("shape:", embedding.weight.shape)
