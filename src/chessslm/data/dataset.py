import torch
from torch.utils.data import Dataset

from chessslm.data.examples import TrainingExample
from chessslm.representation.move import encode_promotion


class ChessDataset(Dataset):
    def __init__(self, examples: list[TrainingExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        example = self.examples[index]

        return {
            "board": torch.tensor(example.board, dtype=torch.long),
            "from_square": torch.tensor(example.from_square, dtype=torch.long),
            "to_square": torch.tensor(example.to_square, dtype=torch.long),
            "promotion": torch.tensor(
                encode_promotion(example.promotion), dtype=torch.long
            ),
            "side_to_move": torch.tensor(example.side_to_move, dtype=torch.long),
            "white_kingside_castling": torch.tensor(
                example.white_kingside_castling,
                dtype=torch.long,
            ),
            "white_queenside_castling": torch.tensor(
                example.white_queenside_castling,
                dtype=torch.long,
            ),
            "black_kingside_castling": torch.tensor(
                example.black_kingside_castling,
                dtype=torch.long,
            ),
            "black_queenside_castling": torch.tensor(
                example.black_queenside_castling,
                dtype=torch.long,
            ),
        }
