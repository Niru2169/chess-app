from __future__ import annotations

"""PGN generation for finished (or in-progress) games.

The SAN movetext is rebuilt with python-chess from the stored UCI moves, so the
export is correct even though the live engine loop only records UCI.
"""

import chess
import chess.pgn

from ..models import EngineConfig, EngineType, Game, Side, STARTPOS_FEN
from .rating import engine_player_name


def _side_name(cfg: EngineConfig | None, side: str) -> str:
    label = "White" if side == Side.WHITE else "Black"
    if cfg is None:
        return label
    if cfg.player_id:
        return cfg.player.name
    if cfg.engine_type == EngineType.PLAYER:
        return f"Human ({label})"
    return engine_player_name(cfg)


def is_pvp_game(game: Game) -> bool:
    """True when both sides are human players (local hot-seat match)."""
    cfgs = list(game.engines.all())
    return len(cfgs) == 2 and all(c.engine_type == EngineType.PLAYER for c in cfgs)


def build_pgn(game: Game) -> str:
    """Build a PGN string for a game from its stored moves.

    Headers include the participants, result and date. The movetext is emitted in
    SAN by replaying the stored UCI moves on a python-chess board.
    """
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "AutoChess"
    pgn_game.headers["Site"] = "AutoChess Arena"

    when = game.finished_at or game.created_at
    pgn_game.headers["Date"] = when.strftime("%Y.%m.%d") if when is not None else "????.??.??"
    pgn_game.headers["Round"] = str(game.pk)

    cfgs = {c.side: c for c in game.engines.all()}
    pgn_game.headers["White"] = _side_name(cfgs.get(Side.WHITE), Side.WHITE)
    pgn_game.headers["Black"] = _side_name(cfgs.get(Side.BLACK), Side.BLACK)

    result = game.result or "*"
    pgn_game.headers["Result"] = result

    if game.termination_reason:
        pgn_game.headers["Termination"] = str(game.termination_reason)

    board = chess.Board(STARTPOS_FEN)
    node = pgn_game
    for move_row in game.moves.order_by("ply_index"):
        try:
            move = chess.Move.from_uci(move_row.uci)
        except (ValueError, TypeError):
            break
        if move not in board.legal_moves:
            promotion = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promotion in board.legal_moves:
                move = promotion
            else:
                break
        node = node.add_variation(move)
        board.push(move)

    pgn_game.headers["Result"] = result
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return pgn_game.accept(exporter)
