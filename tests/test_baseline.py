import torch

from chessslm.models.baseline import ChessBaseline


def test_baseline_output_shapes() -> None:
    model = ChessBaseline()

    board = torch.zeros(
        (4, 64),
        dtype=torch.long,
    )

    from_logits, to_logits, promotion_logits = model(board)

    assert from_logits.shape == torch.Size([4, 64])
    assert to_logits.shape == torch.Size([4, 64])
    assert promotion_logits.shape == torch.Size([4, 5])
