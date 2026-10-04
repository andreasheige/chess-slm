import torch

from chessslm.models.baseline import ChessBaseline


def test_baseline_output_shapes() -> None:
    """The model should return one logit vector per prediction head."""

    model = ChessBaseline()

    # Four chess positions in one batch.
    board = torch.zeros(
        (4, 64),
        dtype=torch.long,
    )

    # 0 = White to move, 1 = Black to move.
    side_to_move = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )

    # Castling rights for each position in the batch.
    #
    # 1 = the right exists
    # 0 = the right has been lost
    white_kingside_castling = torch.tensor(
        [1, 1, 0, 0],
        dtype=torch.long,
    )

    white_queenside_castling = torch.tensor(
        [1, 0, 0, 0],
        dtype=torch.long,
    )

    black_kingside_castling = torch.tensor(
        [1, 1, 1, 0],
        dtype=torch.long,
    )

    black_queenside_castling = torch.tensor(
        [1, 1, 0, 0],
        dtype=torch.long,
    )

    # Teacher-forced source square used by the destination head.
    from_square = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    from_logits, to_logits, promotion_logits = model(
        board,
        side_to_move,
        white_kingside_castling,
        white_queenside_castling,
        black_kingside_castling,
        black_queenside_castling,
        from_square,
    )

    # One class score per possible source square.
    assert from_logits.shape == (4, 64)

    # One class score per possible destination square.
    assert to_logits.shape == (4, 64)

    # Five promotion classes.
    assert promotion_logits.shape == (4, 5)
