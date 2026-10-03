import torch
from torch import nn


class ChessBaseline(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        # Learn a small vector representation for each board token:
        # empty + 12 piece/color combinations.
        self.embedding = nn.Embedding(
            num_embeddings=13,
            embedding_dim=embedding_dim,
        )

        # Convert the complete embedded board into one shared
        # representation of the position.
        self.hidden = nn.Linear(
            64 * embedding_dim,
            hidden_dim,
        )

        # Predict which square the move starts from using only
        # the shared board representation.
        self.from_head = nn.Linear(
            hidden_dim,
            64,
        )

        # Represent the selected source square separately.
        #
        # There are 64 possible squares. The embedding lets the model
        # learn a representation for "the move starts from square X".
        self.from_square_embedding = nn.Embedding(
            num_embeddings=64,
            embedding_dim=embedding_dim,
        )

        # The destination is conditioned on BOTH:
        #
        # 1. the board position
        # 2. the source square
        #
        # Therefore the input is:
        # hidden_dim + embedding_dim
        self.to_head = nn.Linear(
            hidden_dim + embedding_dim,
            64,
        )

        # Promotion still depends only on the shared representation
        # for now. We can revisit this later if the data tells us to.
        self.promotion_head = nn.Linear(
            hidden_dim,
            5,
        )

    def forward(
        self,
        board: torch.Tensor,
        from_square: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, 64]
        #      ↓
        # [B, 64, embedding_dim]
        embedded = self.embedding(board)

        # Flatten the complete board representation:
        #
        # [B, 64, embedding_dim]
        #      ↓
        # [B, 64 * embedding_dim]
        flattened = embedded.flatten(start_dim=1)

        # Build one shared representation of the board.
        hidden = self.hidden(flattened)
        hidden = torch.relu(hidden)

        # Source-square prediction:
        #
        # P(from | board)
        from_logits = self.from_head(hidden)

        # During training we provide the correct source square.
        # This is teacher forcing.
        #
        # [B]
        #  ↓
        # [B, embedding_dim]
        from_embedded = self.from_square_embedding(from_square)

        # Give the destination head both pieces of information:
        #
        # board representation + known source square
        #
        # [B, hidden_dim] + [B, embedding_dim]
        #              ↓
        # [B, hidden_dim + embedding_dim]
        to_input = torch.cat(
            [hidden, from_embedded],
            dim=1,
        )

        # Destination prediction:
        #
        # P(to | board, from)
        to_logits = self.to_head(to_input)

        promotion_logits = self.promotion_head(hidden)

        return (
            from_logits,
            to_logits,
            promotion_logits,
        )
