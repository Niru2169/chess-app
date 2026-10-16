from __future__ import annotations

import threading
import time
import random

import chess
import chess.polyglot
from dataclasses import dataclass
from django.conf import settings

from .uci import UciProcess, UciError, UciBestMove, DisplayInfo
from ..models import EngineConfig, StrengthMode, EngineType
from . import gnn

@dataclass(frozen=True)
class EngineRequest:
    fen: str
    movetime_ms: int
    strength_mode: str
    strength_value: int

class EnginePool:
    """Simple per-process engine pool.

    Note:
        This is fine for the MVP (1 worker). With multiple workers the pool cannot
        be shared across processes, so a separate worker model is recommended when scaling.
    """

    def __init__(self, exe_path: str) -> None:
        self._exe_path = exe_path
        self._lock = threading.Lock()
        self._engine: UciProcess | None = None

    def get(self) -> UciProcess:
        with self._lock:
            if self._engine is None:
                self._engine = UciProcess(self._exe_path)
                self._engine.start()
                self._engine.set_option("Threads", 1)
            return self._engine

_POOL: EnginePool | None = None
_POOL_LOCK = threading.Lock()

def pool() -> EnginePool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = EnginePool(getattr(settings, "AUTOCHESS_STOCKFISH_PATH", "stockfish"))
        return _POOL

def configure_engine(engine: UciProcess, cfg: EngineConfig) -> None:
    """Set the engine strength.

    Args:
        engine: UCI process.
        cfg: Engine config loaded from the DB.

    Raises:
        UciError: If the engine is unavailable.
    """
    # Reset per-move to avoid cross-game leakage:
    engine.new_game()
    # Strength mapping per spec: 1..20 -> Stockfish Skill Level 
    if cfg.strength_mode == StrengthMode.SKILL:
        engine.set_option("UCI_LimitStrength", False)
        engine.set_option("Skill Level", int(cfg.strength_value))
    elif cfg.strength_mode == StrengthMode.ELO:
        engine.set_option("UCI_LimitStrength", True)
        engine.set_option("UCI_Elo", int(cfg.strength_value))
    # Any extra UCI options
    for k, v in (cfg.uci_options or {}).items():
        engine.set_option(str(k), v)

# -----------------------------
# PyChess (python-chess) engine
# -----------------------------

# NOTE: This is a lightweight, self-contained chess engine (negamax + alpha-beta),
# intended for demo/arena use (not for competitive play).
# Strength is approximated via search depth and a modest evaluation function.

MATE_SCORE = 10_000_000

_PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

def _dist_to_center(file: int, rank: int) -> float:
    # center is between 3 and 4
    return abs(file - 3.5) + abs(rank - 3.5)

def _chebyshev(a_file: int, a_rank: int, b_file: int, b_rank: int) -> int:
    return max(abs(a_file - b_file), abs(a_rank - b_rank))

def _endgame_phase(board: chess.Board) -> float:
    """0.0 = opening/middlegame, 1.0 = deep endgame."""
    non_pawn = 0
    for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        non_pawn += len(board.pieces(p, chess.WHITE)) * _PIECE_VALUE[p]
        non_pawn += len(board.pieces(p, chess.BLACK)) * _PIECE_VALUE[p]
    # Rough scale: start position ~ 2*(2N+2B+2R+Q)= 2*(640+660+1000+900)= 6400
    return max(0.0, min(1.0, (6400 - non_pawn) / 6400.0))

def _positional_bonus(board: chess.Board, color: bool) -> int:
    """Simple positional heuristics (center, development, king safety/endgame activity)."""
    bonus = 0
    phase = _endgame_phase(board)

    # King square (blend between safety and activity)
    ksq = next(iter(board.pieces(chess.KING, color)), None)
    if ksq is not None:
        f = chess.square_file(ksq)
        r = chess.square_rank(ksq)
        # Endgame: towards center
        endgame = int((7.0 - _dist_to_center(f, r)) * 12)
        # Middlegame: towards corners (castling-ish)
        corners = [(0, 0), (7, 0), (0, 7), (7, 7)]
        dcorner = min(_chebyshev(f, r, cf, cr) for cf, cr in corners)
        midgame = int((7 - dcorner) * 10) - int(_dist_to_center(f, r) * 6)
        bonus += int(phase * endgame + (1.0 - phase) * midgame)

    # Pawns: advancement + mild centralization
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        rr = r if color == chess.WHITE else (7 - r)
        bonus += rr * 6
        bonus += int((3.5 - abs(f - 3.5)) * 2)

    # Knights/Bishops/Queen: prefer central squares
    for sq in board.pieces(chess.KNIGHT, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        bonus += int((7.0 - _dist_to_center(f, r)) * 10)
    for sq in board.pieces(chess.BISHOP, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        bonus += int((7.0 - _dist_to_center(f, r)) * 6)
    for sq in board.pieces(chess.QUEEN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        bonus += int((7.0 - _dist_to_center(f, r)) * 2)

    # Rooks: prefer open/semi-open files and 7th rank
    pawns_w = board.pieces(chess.PAWN, chess.WHITE)
    pawns_b = board.pieces(chess.PAWN, chess.BLACK)
    for sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        has_own_pawn = any(chess.square_file(psq) == f for psq in (pawns_w if color == chess.WHITE else pawns_b))
        has_enemy_pawn = any(chess.square_file(psq) == f for psq in (pawns_b if color == chess.WHITE else pawns_w))
        if not has_own_pawn and not has_enemy_pawn:
            bonus += 18  # open file
        elif not has_own_pawn and has_enemy_pawn:
            bonus += 10  # semi-open
        rr = r if color == chess.WHITE else (7 - r)
        if rr == 6:
            bonus += 12  # 7th rank

    # Bishop pair
    if len(board.pieces(chess.BISHOP, color)) >= 2:
        bonus += 18

    return bonus

def _evaluate_white_pov(board: chess.Board) -> int:
    """Evaluation from White's perspective (positive = good for White)."""
    # Terminal positions
    if board.is_checkmate():
        # Side to move is checkmated
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Material
    w = 0
    b = 0
    for p, v in _PIECE_VALUE.items():
        w += len(board.pieces(p, chess.WHITE)) * v
        b += len(board.pieces(p, chess.BLACK)) * v
    score = w - b

    # Positional
    score += _positional_bonus(board, chess.WHITE)
    score -= _positional_bonus(board, chess.BLACK)

    # Mobility (cheap-ish count)
    try:
        mob = board.legal_moves.count()
    except Exception:
        mob = len(list(board.legal_moves))
    # Small tempo/mobility bonus for side to move
    score += (mob * 2) if board.turn == chess.WHITE else -(mob * 2)

    # Slight penalty for being in check (encourages getting out of check)
    if board.is_check():
        score += -20 if board.turn == chess.WHITE else 20

    return score

def _evaluate(board: chess.Board) -> int:
    """Evaluation from side-to-move perspective (negamax-friendly)."""
    s = _evaluate_white_pov(board)
    return s if board.turn == chess.WHITE else -s

def _order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Move ordering: MVV-LVA-ish captures and checks first."""
    def score(m: chess.Move) -> int:
        sc = 0
        if board.is_capture(m):
            # Captures first; approximate victim value
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            if victim:
                sc += 1000 + _PIECE_VALUE.get(victim.piece_type, 0)
            if attacker:
                sc -= _PIECE_VALUE.get(attacker.piece_type, 0) // 10
        if m.promotion:
            sc += 900
        board.push(m)
        if board.is_check():
            sc += 60
        board.pop()
        return -sc
    return sorted(moves, key=score)

def _quiesce(board: chess.Board, alpha: int, beta: int, end_t: float) -> int:
    if time.monotonic() >= end_t:
        return _evaluate(board)

    stand_pat = _evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Only consider captures in qsearch (keeps it fast)
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    for mv in _order_moves(board, captures):
        if time.monotonic() >= end_t:
            break
        board.push(mv)
        score = -_quiesce(board, -beta, -alpha, end_t)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, end_t: float, ply: int) -> int:
    if time.monotonic() >= end_t:
        return _evaluate(board)

    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate():
        return 0

    if depth <= 0:
        return _quiesce(board, alpha, beta, end_t)

    moves = list(board.legal_moves)
    if not moves:
        return -MATE_SCORE + ply if board.is_check() else 0

    best = -MATE_SCORE
    for mv in _order_moves(board, moves):
        if time.monotonic() >= end_t:
            break
        board.push(mv)
        val = -_negamax(board, depth - 1, -beta, -alpha, end_t, ply + 1)
        board.pop()
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best

def _depth_from_strength(cfg: EngineConfig) -> int:
    # Base depth by configured strength
    if cfg.strength_mode == StrengthMode.SKILL:
        s = int(cfg.strength_value)
        if s <= 3:
            d = 1
        elif s <= 6:
            d = 2
        elif s <= 10:
            d = 3
        elif s <= 14:
            d = 4
        elif s <= 17:
            d = 5
        else:
            d = 6
    else:
        e = int(cfg.strength_value)
        if e < 1100:
            d = 2
        elif e < 1500:
            d = 3
        elif e < 1900:
            d = 4
        elif e < 2300:
            d = 5
        else:
            d = 6

    # Give a little extra depth with larger movetime budgets
    mt = int(getattr(cfg, "movetime_ms", 150) or 150)
    if mt >= 500:
        d += 1
    if mt >= 1000:
        d += 1

    return max(1, min(7, d))

def _pychess_choose_move(fen: str, cfg: EngineConfig) -> str:
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "(none)"

    target_depth = _depth_from_strength(cfg)

    # Time budget with safety margin
    budget_s = max(0.02, float(cfg.movetime_ms) / 1000.0)
    end_t = time.monotonic() + budget_s * 0.92

    # At low skill, intentionally add randomness among decent candidates.
    noise = 0.0
    if cfg.strength_mode == StrengthMode.SKILL:
        s = int(cfg.strength_value)
        if s <= 6:
            noise = 0.35
        elif s <= 10:
            noise = 0.15

    ordered = _order_moves(board, moves)
    best_move = ordered[0]
    best_score = -MATE_SCORE

    # Iterative deepening gives better results under a hard time limit.
    for depth in range(1, target_depth + 1):
        if time.monotonic() >= end_t:
            break

        cur_best_move = best_move
        cur_best_score = -MATE_SCORE

        # Search each candidate move at this depth
        # (use wide alpha/beta; per-move aspiration windows are overkill here)
        for mv in ordered:
            if time.monotonic() >= end_t:
                break
            board.push(mv)
            val = -_negamax(board, depth - 1, -MATE_SCORE, MATE_SCORE, end_t, ply=1)
            board.pop()

            if noise:
                val = int(val + random.uniform(-1.0, 1.0) * 200.0 * noise)

            if val > cur_best_score:
                cur_best_score = val
                cur_best_move = mv

        # If we completed at least one move at this depth, accept the best result.
        best_move = cur_best_move
        best_score = cur_best_score

        # PV move first in next iteration (helps ordering a bit)
        ordered = [best_move] + [m for m in ordered if m != best_move]

    return best_move.uci()



def best_move(fen: str, cfg: EngineConfig) -> UciBestMove:
    # Engine switch: STOCKFISH (UCI) vs PYCHESS (python-chess minimax) vs GNN (BayesianELO)
    engine_type = getattr(cfg, 'engine_type', EngineType.STOCKFISH)
    if engine_type == EngineType.PYCHESS:
        mv = _pychess_choose_move(fen, cfg)
        return UciBestMove(move=mv)
    if engine_type == EngineType.GNN:
        try:
            mv = gnn.best_move(fen, depth=getattr(cfg, "search_depth", 0) or 0)
        except gnn.GnnUnavailable as exc:
            raise UciError(f"GNN engine unavailable: {exc}") from exc
        return UciBestMove(move=mv)
    engine = pool().get()
    configure_engine(engine, cfg)
    engine.position_fen(fen)
    engine.is_ready(timeout_s=2.0)
    return engine.go_movetime(int(cfg.movetime_ms), timeout_s=max(3.0, cfg.movetime_ms / 1000.0 + 2.0))


def _display_info_from_board(board: chess.Board) -> DisplayInfo:
    """Describe a position using python-chess (Stockfish-free fallback)."""
    checkers = board.checkers()
    checkers_raw = " ".join(chess.square_name(sq) for sq in sorted(checkers)) if checkers else ""
    try:
        key = str(chess.polyglot.zobrist_hash(board))
    except Exception:
        key = ""
    return DisplayInfo(fen=board.fen(), key=key, checkers_raw=checkers_raw)


def apply_and_inspect(fen_before: str, uci_move: str) -> DisplayInfo:
    """Apply ``uci_move`` and describe the resulting position.

    Prefers Stockfish's non-standard 'd' command (Fen/Key/Checkers); falls back
    to python-chess when no Stockfish binary is available so that games between
    the built-in engines still run without an external binary.

    Raises:
        UciError: If the move is illegal or cannot be parsed.
    """
    board = chess.Board(fen_before)
    try:
        move = chess.Move.from_uci(uci_move)
    except ValueError as exc:
        raise UciError(f"Invalid move: {uci_move!r}") from exc
    if move not in board.legal_moves:
        raise UciError(f"Illegal move: {uci_move!r}")

    try:
        engine = pool().get()
        engine.new_game()
        engine.position_fen(fen_before, moves=[uci_move])
        engine.is_ready(timeout_s=2.0)
        return engine.display(timeout_s=2.0)
    except (UciError, OSError):
        board.push(move)
        return _display_info_from_board(board)


def inspect_position(fen: str) -> DisplayInfo:
    try:
        engine = pool().get()
        engine.new_game()
        engine.position_fen(fen)
        engine.is_ready(timeout_s=2.0)
        return engine.display(timeout_s=2.0)
    except (UciError, OSError):
        return _display_info_from_board(chess.Board(fen))
