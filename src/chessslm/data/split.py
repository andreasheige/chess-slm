from random import Random

import chess.pgn

from chessslm.data.examples import TrainingExample


def split_examples(
    examples: list[TrainingExample],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split examples into reproducible training and validation sets.

    The training set is used to update model parameters.

    The validation set is never used for optimizer updates. It is only
    used to measure how well the model performs on examples it did not
    train on.

    A fixed random seed makes the split reproducible, which is important
    when comparing experiments.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    shuffled = examples.copy()

    random = Random(seed)
    random.shuffle(shuffled)

    validation_size = max(
        1,
        round(len(shuffled) * validation_fraction),
    )

    validation_examples = shuffled[:validation_size]
    training_examples = shuffled[validation_size:]

    return training_examples, validation_examples


def split_games(
    games: list[chess.pgn.Game],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[chess.pgn.Game], list[chess.pgn.Game]]:
    """Split complete games into training and validation sets.

    Games are split before they are converted into individual position
    examples. This prevents positions from the same game appearing in
    both training and validation data.

    Keeping complete games together gives us a more meaningful
    validation set when training on multiple PGN games.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    shuffled = games.copy()

    random = Random(seed)
    random.shuffle(shuffled)

    validation_size = max(
        1,
        round(len(shuffled) * validation_fraction),
    )

    validation_games = shuffled[:validation_size]
    training_games = shuffled[validation_size:]

    return training_games, validation_games
