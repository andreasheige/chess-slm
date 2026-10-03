from pathlib import Path

import chess.pgn


def load_games(
    path: Path,
) -> list[chess.pgn.Game]:
    """Load every chess game from a PGN file.

    A PGN file can contain multiple games. python-chess reads one game
    at a time, so we continue reading until the end of the file.

    Keeping PGN loading separate from example creation gives us two
    clear responsibilities:

    PGN file -> games
    games    -> training examples
    """

    games: list[chess.pgn.Game] = []

    with path.open() as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)

            # read_game() returns None when there are no more games
            # left in the PGN file.
            if game is None:
                break

            games.append(game)

    return games
