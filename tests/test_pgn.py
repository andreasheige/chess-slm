from pathlib import Path

from chessslm.data.pgn import load_games


def test_load_games_reads_multiple_games(
    tmp_path: Path,
) -> None:
    pgn_path = tmp_path / "games.pgn"

    pgn_path.write_text(
        """
[Event "Game 1"]
[Result "*"]

1. e4 e5 *

[Event "Game 2"]
[Result "*"]

1. d4 d5 *
""".strip()
    )

    games = load_games(pgn_path)

    assert len(games) == 2
    assert games[0].headers["Event"] == "Game 1"
    assert games[1].headers["Event"] == "Game 2"
