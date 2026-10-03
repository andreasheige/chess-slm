import chess
import torch
from torch import nn

from chessslm.models.baseline import ChessBaseline
from chessslm.representation.board import encode_board

model = ChessBaseline()
loss_fn = nn.CrossEntropyLoss()

# One real training example:
# initial chess position -> e2e4
chess_board = chess.Board()
encoded_board = encode_board(chess_board)

board = torch.tensor(
    [encoded_board],
    dtype=torch.long,
)
target_from = torch.tensor([chess.E2], dtype=torch.long)

# Forward pass
from_logits, _, _ = model(board)

# Measure error
loss = loss_fn(from_logits, target_from)

print("Board shape:", board.shape)
print("From logits shape:", from_logits.shape)
print("Target:", target_from)
print("Loss:", loss.item())

print("\nEmbedding gradients before backward:")
print(model.embedding.weight.grad)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)
before = model.embedding.weight.detach().clone()
# Backpropagation
loss.backward()
optimizer.step()
after = model.embedding.weight.detach().clone()
print("\nWeights changed:")
print(torch.equal(before, after))

print("\nFirst embedding row before:")
print(before[0])

print("\nFirst embedding row after:")
print(after[0])
print("\nEmbedding gradients after backward:")
print(model.embedding.weight.grad)
