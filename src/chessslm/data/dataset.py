from torch.utils.data import Dataset

from chessslm.data.examples import TrainingExample


class ChessDataset(Dataset):
    def __init__(self, examples: list[TrainingExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]
