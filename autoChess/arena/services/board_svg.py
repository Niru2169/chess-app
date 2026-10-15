from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess
import chess.svg


# Color palette: a "brown" board matching the python-chess docs/ examples.
# (The light/dark squares approximate the bundled reference image.)
DEFAULT_COLORS = {
    "square light": "#f0d9b5",
    "square dark": "#b58863",
    # Last-move highlight (on light/dark squares).
    "square light lastmove": "#f6f595",
    "square dark lastmove": "#bdc959",
    # No coordinates/margin needed: our own frame provides them.
    "margin": "#ffffff00",
    "coord": "#00000000",
    "inner border": "#00000000",
    "outer border": "#00000000",
}


def render_board_svg(fen: str, last_move_uci: str | None = None, *, size: int = 560) -> str:
    """Render a position as SVG using python-chess.

    - No external image files: the pieces come from the python-chess `chess.svg` module.
    - The template draws the coordinates (left + bottom), hence `coordinates=False`.

    Args:
        fen: Current FEN.
        last_move_uci: Last move in UCI (e.g. e2e4). Optional.
        size: SVG size in pixels (CSS may override it responsively).

    Returns:
        SVG markup string.
    """
    board = chess.Board(fen)

    lastmove = None
    if last_move_uci:
        try:
            lastmove = chess.Move.from_uci(last_move_uci)
        except ValueError:
            lastmove = None

    check_sq = None
    try:
        if board.is_check():
            # The king of the side to move is in check.
            king_sq = board.king(board.turn)
            check_sq = king_sq
    except Exception:
        check_sq = None

    svg = chess.svg.board(
        board=board,
        size=size,
        coordinates=False,
        lastmove=lastmove,
        check=check_sq,
        colors=DEFAULT_COLORS,
        borders=False,
    )
    return svg
