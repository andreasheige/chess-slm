from random import Random

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
