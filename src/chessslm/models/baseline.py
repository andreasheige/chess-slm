import torch
from torch import nn


class ChessBaseline(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=13,
            embedding_dim=embedding_dim,
        )

        self.hidden = nn.Linear(
            64 * embedding_dim,
            hidden_dim,
        )

        self.from_head = nn.Linear(hidden_dim, 64)
        self.to_head = nn.Linear(hidden_dim, 64)
        self.promotion_head = nn.Linear(hidden_dim, 5)

    def forward(
        self,
        board: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(board)

        flattened = embedded.flatten(start_dim=1)

        hidden = self.hidden(flattened)
        hidden = torch.relu(hidden)

        from_logits = self.from_head(hidden)
        to_logits = self.to_head(hidden)
        promotion_logits = self.promotion_head(hidden)

        return from_logits, to_logits, promotion_logits
