import chess
import torch
from torch import nn

from chessslm.models.baseline import ChessBaseline
from chessslm.representation.board import encode_board

torch.manual_seed(42)

model = ChessBaseline()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)

chess_board = chess.Board()
encoded_board = encode_board(chess_board)

board = torch.tensor(
    [encoded_board],
    dtype=torch.long,
)

target_from = torch.tensor([chess.E2], dtype=torch.long)
target_to = torch.tensor([chess.E4], dtype=torch.long)
target_promotion = torch.tensor([0], dtype=torch.long)

for step in range(101):
    optimizer.zero_grad()

    from_logits, to_logits, promotion_logits = model(board)

    from_loss = loss_fn(from_logits, target_from)
    to_loss = loss_fn(to_logits, target_to)
    promotion_loss = loss_fn(
        promotion_logits,
        target_promotion,
    )

    loss = from_loss + to_loss + promotion_loss

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        from_prediction = torch.argmax(from_logits, dim=1)
        to_prediction = torch.argmax(to_logits, dim=1)
        promotion_prediction = torch.argmax(
            promotion_logits,
            dim=1,
        )

        print(
            f"step={step:3d} "
            f"loss={loss.item():.4f} "
            f"move={from_prediction.item()}→{to_prediction.item()} "
            f"promotion={promotion_prediction.item()}"
        )
