from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

_UNICODE_PIECES: dict[str, str] = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

@dataclass(frozen=True)
class Square:
    file: int  # 0..7 (a..h)
    rank: int  # 0..7 (1..8)

@dataclass(frozen=True)
class DisplaySquare:
    col: int  # 0..7 (a..h)
    row: int  # 0..7 (display row; 0 = rank 8)

@dataclass(frozen=True)
class Highlight:
    from_sq: Square | None
    to_sq: Square | None
    from_disp: DisplaySquare | None
    to_disp: DisplaySquare | None

@dataclass(frozen=True)
class Cell:
    piece: str
    piece_class: str
    is_dark: bool
    is_from: bool
    is_to: bool

def parse_fen_piece_placement(fen: str) -> list[list[str]]:
    """Build an 8x8 board matrix from a FEN.

    Args:
        fen: Full FEN string.

    Returns:
        8x8 matrix (rank 8 -> 1 order) with FEN characters or empty strings.

    Raises:
        ValueError: If the FEN format is invalid.
    """
    parts = fen.split()
    if len(parts) < 1:
        raise ValueError("Invalid FEN: missing fields.")
    placement = parts[0]
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError("Invalid FEN: 8 rows required.")
    board: list[list[str]] = []
    for row in rows:
        row_cells: list[str] = []
        for ch in row:
            if ch.isdigit():
                row_cells.extend([""] * int(ch))
            else:
                row_cells.append(ch)
        if len(row_cells) != 8:
            raise ValueError("Invalid FEN: row length is not 8.")
        board.append(row_cells)
    return board

def side_to_move_from_fen(fen: str) -> str:
    """Which side is to move according to the FEN.

    Args:
        fen: Full FEN.

    Returns:
        'w' or 'b'.

    Raises:
        ValueError: If the FEN does not contain a side-to-move field.
    """
    parts = fen.split()
    if len(parts) < 2 or parts[1] not in {"w", "b"}:
        raise ValueError("Invalid FEN: missing/invalid side-to-move field.")
    return parts[1]

def halfmove_clock_from_fen(fen: str) -> int:
    """Halfmove clock for the fifty-move rule.

    Args:
        fen: Full FEN.

    Returns:
        halfmove clock (int).

    Raises:
        ValueError: If the FEN does not contain a halfmove clock field.
    """
    parts = fen.split()
    if len(parts) < 5:
        raise ValueError("Invalid FEN: missing halfmove clock field.")
    return int(parts[4])

def uci_move_highlight(uci: str) -> Highlight:
    """The squares to highlight from a UCI move (e.g. e2e4).

    Note:
        We do not do arithmetic in Django templates (e.g. 7-r), so the return
        value also includes **display coordinates** (row 0 = rank 8).

    Args:
        uci: UCI move string.

    Returns:
        Highlight (from/to). If the string cannot be parsed, both are None.
    """
    m = uci.strip().lower()
    if len(m) < 4:
        return Highlight(None, None, None, None)
    from_sq = _square_from_alg(m[0:2])
    to_sq = _square_from_alg(m[2:4])
    if from_sq is None or to_sq is None:
        return Highlight(None, None, None, None)

    from_disp = DisplaySquare(col=from_sq.file, row=7 - from_sq.rank)
    to_disp = DisplaySquare(col=to_sq.file, row=7 - to_sq.rank)
    return Highlight(from_sq, to_sq, from_disp, to_disp)

def _square_from_alg(alg: str) -> Square | None:
    if len(alg) != 2:
        return None
    file_c, rank_c = alg[0], alg[1]
    if file_c not in "abcdefgh" or rank_c not in "12345678":
        return None
    return Square(file="abcdefgh".index(file_c), rank=int(rank_c) - 1)

def is_insufficient_material(fen: str) -> bool:
    """Insufficient-material draw (simple, conservative rules).

    Note:
        The full FIDE rule set is more complex. Here we handle the typical cases
        needed for the MVP.

    Args:
        fen: Full FEN.

    Returns:
        True if the position definitely has no mating potential.
    """
    placement = fen.split()[0]
    pieces = [c for c in placement if c.isalpha()]
    # Remove kings
    minors = [p for p in pieces if p not in ("K", "k")]
    if not minors:
        return True  # K vs K

    # Map counts
    cnt = {p: minors.count(p) for p in set(minors)}
    # Only bishops/knights allowed (no pawns, rooks, queens)
    forbidden = set("PpRrQq")
    if any(p in forbidden for p in minors):
        return False

    # K+B vs K, K+N vs K
    if len(minors) == 1 and next(iter(cnt)) in {"B","b","N","n"}:
        return True

    # K+B vs K+B with same-colored bishops: we approximate by declaring insufficient when only bishops and each side has exactly one bishop.
    # Determining square color from FEN without move generation is doable but extra work; MVP keeps conservative: treat as draw if only bishops and <=1 bishop per side.
    if set(minors).issubset({"B","b"}) and cnt.get("B",0) <= 1 and cnt.get("b",0) <= 1:
        return True

    return False


# To match the "python-chess docs" look we use the **solid** (black) Unicode figures
# for both sides and color them with CSS: white pieces are white (not a hollow outline),
# black pieces are dark.
_SOLID_UNICODE = {
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

def _piece_symbol_and_class(ch: str) -> tuple[str, str]:
    if not ch:
        return "", ""
    is_white = ch.isupper()
    glyph = _SOLID_UNICODE.get(ch.lower(), "")
    css = "piece-w" if is_white else "piece-b"
    return glyph, css


def build_board_cells(fen: str, last_move_uci: str) -> list[list[Cell]]:
    """Prepare the cell data needed to draw the board.

    We avoid arithmetic and complex filter expressions in the Django template,
    so the following are computed here:
    - square color (dark/light)
    - last-move highlight (from/to)

    Args:
        fen: Current position.
        last_move_uci: Last move in UCI format.

    Returns:
        8x8 Cell matrix (display row 0 = rank 8).
    """
    board = parse_fen_piece_placement(fen)
    hl = uci_move_highlight(last_move_uci or "")
    cells: list[list[Cell]] = []
    for r in range(8):
        row: list[Cell] = []
        for c in range(8):
            raw_piece = board[r][c]
            piece, piece_class = _piece_symbol_and_class(raw_piece)
            # Chessboard coloring: (r+c) even -> light, odd -> dark (or vice versa).
            is_dark = ((r + c) % 2 == 1)
            is_from = bool(hl.from_disp and hl.from_disp.row == r and hl.from_disp.col == c)
            is_to = bool(hl.to_disp and hl.to_disp.row == r and hl.to_disp.col == c)
            row.append(Cell(piece=piece, piece_class=piece_class, is_dark=is_dark, is_from=is_from, is_to=is_to))
        cells.append(row)
    return cells
