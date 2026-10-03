from dataclasses import dataclass

import chess
import chess.pgn

from chessslm.representation.board import encode_board
from chessslm.representation.move import encode_move


@dataclass(frozen=True)
class TrainingExample:
    board: list[int]
    from_square: int
    to_square: int
    promotion: int | None


def create_training_example(
    board: chess.Board,
    move: chess.Move,
) -> TrainingExample:
    encoded_board = encode_board(board)
    from_square, to_square, promotion = encode_move(move)

    return TrainingExample(
        board=encoded_board,
        from_square=from_square,
        to_square=to_square,
        promotion=promotion,
    )


def create_examples_from_game(game: chess.pgn.Game) -> list[TrainingExample]:
    board = game.board()
    examples: list[TrainingExample] = []

    for move in game.mainline_moves():
        example = create_training_example(board, move)
        examples.append(example)
        board.push(move)

    return examples


def create_examples_from_games(
    games: list[chess.pgn.Game],
) -> list[TrainingExample]:
    """Create training examples from multiple chess games.

    Each game is converted independently into position -> move examples.
    The resulting examples are then combined into one flat list that can
    be passed to ChessDataset.

    Game-level train/validation splitting should happen BEFORE calling
    this function so examples from one game never leak across the split.
    """

    examples: list[TrainingExample] = []

    for game in games:
        examples.extend(create_examples_from_game(game))

    return examples
