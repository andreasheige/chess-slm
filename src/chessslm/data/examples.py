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
