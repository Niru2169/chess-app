from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import chess

from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from ..models import (
    Game, EngineConfig, Side, GameStatus, TerminationReason, Move, MatchRecord, StrengthMode, EngineType, STARTPOS_FEN
)
from .fen import side_to_move_from_fen, halfmove_clock_from_fen, is_insufficient_material
from .engine import best_move, apply_and_inspect, inspect_position
from .rating import apply_ratings_for_game
from .pgn import build_pgn
from .uci import UciError

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TickOutcome:
    advanced: bool
    message: str

def ensure_engine_rows(game: Game) -> None:
    """Ensure an EngineConfig row exists for both sides.

    Args:
        game: Game instance.

    Returns:
        None
    """
    existing = {e.side: e for e in game.engines.all()}
    if Side.WHITE not in existing:
        EngineConfig.objects.create(
            game=game,
            side=Side.WHITE,
            engine_type=EngineType.PYCHESS,
            strength_mode=StrengthMode.SKILL,
            strength_value=10,
            movetime_ms=getattr(settings, "AUTOCHESS_DEFAULT_MOVETIME_MS", 150),
        )
    if Side.BLACK not in existing:
        EngineConfig.objects.create(
            game=game,
            side=Side.BLACK,
            engine_type=EngineType.STOCKFISH,
            strength_mode=StrengthMode.SKILL,
            strength_value=10,
            movetime_ms=getattr(settings, "AUTOCHESS_DEFAULT_MOVETIME_MS", 150),
        )

def _preview_ms_for_game(game) -> int:
    """Per-game preview time. Falls back to 350ms if not set."""
    try:
        return int(getattr(game, 'preview_ms', 350) or 350)
    except Exception:
        return 350


def tick(game_id: int) -> TickOutcome:
    """One tick: execute at most one ply inside a transaction.

    Two-phase move (preview):
    1) the next move is computed and stored in `pending_move_uci`, then just highlighted;
    2) on one of the following tick(s) (after AUTOCHESS_PREVIEW_MS) the move is actually applied.

    Per the specification the tick is:
    - protected by a row lock (select_for_update),
    - does nothing unless the game is RUNNING,
    - paced by next_action_at + move_interval_ms (the preview time is included in the cycle).

    Args:
        game_id: The match identifier.

    Returns:
        TickOutcome: whether a move was made and a short message.
    """
    with transaction.atomic():
        # Acquire per-game tick lock to avoid double-tick races (multiple tabs) and
        # to reduce SQLite write-lock contention on demo deployments.
        lock_now = timezone.now()
        stale_before = lock_now - timedelta(seconds=60)
        acquired = Game.objects.filter(pk=game_id).filter(
            Q(tick_lock=False) | Q(tick_lock_at__lt=stale_before) | Q(tick_lock_at__isnull=True)
        ).update(tick_lock=True, tick_lock_at=lock_now)
        if acquired == 0:
            return TickOutcome(False, "Tick: busy (locked).")

        try:
                game = Game.objects.select_for_update().get(pk=game_id)
                ensure_engine_rows(game)

                if game.status != GameStatus.RUNNING:
                    return TickOutcome(False, "Not running.")

                now = timezone.now()
                preview_ms = _preview_ms_for_game(game)

                # If there is a pending move and the preview time has not elapsed, just return (keep the highlight).
                if game.pending_move_uci:
                    deadline = (game.pending_move_set_at or now) + timedelta(milliseconds=preview_ms)
                    if now < deadline:
                        return TickOutcome(False, "Preview (move highlighted).")
                    move_uci = game.pending_move_uci
                else:
                    # No pending move: the cycle speed is governed by next_action_at.
                    if game.next_action_at is not None and now < game.next_action_at:
                        return TickOutcome(False, "Waiting (speed).")

                    # Update side_to_move from current FEN (authoritative)
                    try:
                        stm = side_to_move_from_fen(game.fen)
                    except ValueError:
                        game.mark_finished(result="1/2-1/2", reason=TerminationReason.UNKNOWN)
                        game.save(update_fields=["status","result","termination_reason","finished_at","updated_at"])
                        return TickOutcome(False, "Invalid FEN; closed.")

                    game.side_to_move = stm
                    side = Side.WHITE if stm == "w" else Side.BLACK
                    cfg = game.engines.get(side=side)

                    if cfg.engine_type == EngineType.PLAYER:
                        # For a human player the tick does not move; it just waits for UI input.
                        # As a safeguard, clear any pending state that got stuck.
                        if (game.pending_move_uci or "").strip():
                            game.pending_move_uci = ""
                            game.pending_move_set_at = None
                            game.save(update_fields=["pending_move_uci", "pending_move_set_at", "updated_at"])
                        return TickOutcome(False, "Waiting: player to move.")

                    # Ask the engine for a move (but do not apply it yet, only store it as a preview)
                    try:
                        bm = best_move(game.fen, cfg)
                    except UciError:
                        _finish_forfeit(game, loser_side=side)
                        return TickOutcome(False, "Engine error; forfeit.")

                    if bm.move in {"(none)", "0000"}:
                        _finish_no_moves(game)
                        return TickOutcome(False, "No legal move; game over.")

                    game.pending_move_uci = bm.move
                    game.pending_move_set_at = now
                    game.save(update_fields=["pending_move_uci", "pending_move_set_at", "side_to_move", "updated_at"])
                    return TickOutcome(False, "Preview (move highlighted).")

                # Apply & validate legality (fen changed)
                # Update side again for forfeit attribution / robustness
                try:
                    stm = side_to_move_from_fen(game.fen)
                except ValueError:
                    game.mark_finished(result="1/2-1/2", reason=TerminationReason.UNKNOWN)
                    game.pending_move_uci = ""
                    game.pending_move_set_at = None
                    game.save(update_fields=["status","result","termination_reason","finished_at","pending_move_uci","pending_move_set_at","updated_at"])
                    return TickOutcome(False, "Invalid FEN; closed.")

                side = Side.WHITE if stm == "w" else Side.BLACK

                try:
                    info_after = apply_and_inspect(game.fen, move_uci)
                except UciError:
                    _finish_forfeit(game, loser_side=side)
                    return TickOutcome(False, "Engine error; forfeit.")

                fen_after = info_after.fen or ""
                if not fen_after or fen_after == game.fen:
                    _finish_forfeit(game, loser_side=side)
                    return TickOutcome(False, "Illegal move; forfeit.")

                # Determine check on opponent (Checkers in the resulting position means side-to-move is in check)
                is_check = bool((info_after.checkers_raw or "").strip())

                # Persist move
                ply_index = game.ply_count + 1
                try:
                    san = chess.Board(game.fen).san(chess.Move.from_uci(move_uci))
                except Exception:
                    san = move_uci
                Move.objects.create(
                    game=game,
                    ply_index=ply_index,
                    uci=move_uci,
                    san=san,
                    fen_after=fen_after,
                    position_key_after=info_after.key or "",
                    is_check=is_check,
                )

                game.fen = fen_after
                game.ply_count = ply_index
                game.last_move_uci = move_uci
                game.last_move_san = san
                game.side_to_move = side_to_move_from_fen(fen_after)

                # Clear pending preview state
                game.pending_move_uci = ""
                game.pending_move_set_at = None

                # Next action time:
                # move_interval_ms includes the preview time; the remaining wait is:
                remaining_ms = max(0, int(game.move_interval_ms) - preview_ms)
                game.next_action_at = now + timedelta(milliseconds=remaining_ms)

                # End conditions / draw policies
                if ply_index >= int(getattr(settings, "AUTOCHESS_MAX_PLIES", 600)):
                    _finish_draw(game, TerminationReason.MAX_PLIES)
                    return TickOutcome(True, "Max plies draw.")

                # Auto-claimable draws: repetition / fifty-move / insufficient
                if _is_threefold_repetition(game):
                    _finish_draw(game, TerminationReason.REPETITION)
                    return TickOutcome(True, "Threefold repetition: draw.")
                if _is_fifty_move(game):
                    _finish_draw(game, TerminationReason.FIFTY_MOVE)
                    return TickOutcome(True, "Fifty-move rule: draw.")
                if is_insufficient_material(game.fen):
                    _finish_draw(game, TerminationReason.INSUFFICIENT)
                    return TickOutcome(True, "Insufficient material: draw.")

                game.save(update_fields=[
                    "fen","ply_count","last_move_uci","last_move_san","side_to_move",
                    "pending_move_uci","pending_move_set_at","next_action_at","updated_at"
                ])

                return TickOutcome(True, "Move applied.")
        finally:
            # Release tick lock (best effort)
            Game.objects.filter(pk=game_id).update(tick_lock=False)



# ---- finishing helpers ----

def _finish_draw(game: Game, reason: str) -> None:
    game.mark_finished(result="1/2-1/2", reason=reason)
    game.save()
    finalize_game(game)

def _finish_forfeit(game: Game, loser_side: str) -> None:
    winner = "1-0" if loser_side == Side.BLACK else "0-1"
    game.mark_finished(result=winner, reason=TerminationReason.RESIGN)
    game.save()
    finalize_game(game)

def _finish_no_moves(game: Game) -> None:
    # Determine mate vs stalemate: inspect the current position, 'Checkers:' indicates check.
    info = inspect_position(game.fen)
    in_check = bool((info.checkers_raw or "").strip())
    stm = side_to_move_from_fen(game.fen)
    loser_side = Side.WHITE if stm == "w" else Side.BLACK
    if in_check:
        winner = "1-0" if loser_side == Side.BLACK else "0-1"
        game.mark_finished(result=winner, reason=TerminationReason.CHECKMATE)
    else:
        game.mark_finished(result="1/2-1/2", reason=TerminationReason.STALEMATE)
    game.save()
    finalize_game(game)


def finalize_game(game: Game) -> None:
    """Close out a match: MatchRecord + ratings (Elo/BayesElo).

    Idempotent: repeated calls do not double-count the rating update.
    """
    _materialize_match_record(game)
    try:
        apply_ratings_for_game(game)
    except Exception:
        # A rating failure must not prevent the match from being closed.
        logger.exception("Rating update failed (game=%s)", getattr(game, "pk", None))


def _materialize_match_record(game: Game) -> None:
    white = game.engines.get(side=Side.WHITE)
    black = game.engines.get(side=Side.BLACK)
    MatchRecord.objects.update_or_create(
        game=game,
        defaults={
            "result": game.result,
            "termination_reason": game.termination_reason,
            "white_strength_mode": white.strength_mode,
            "white_strength_value": white.strength_value,
            "black_strength_mode": black.strength_mode,
            "black_strength_value": black.strength_value,
            "move_interval_ms": game.move_interval_ms,
            "pgn": build_pgn(game),
            "finished_at": game.finished_at or timezone.now(),
        }
    )

def _is_fifty_move(game: Game) -> bool:
    try:
        return halfmove_clock_from_fen(game.fen) >= 100
    except Exception:
        return False

def _is_threefold_repetition(game: Game) -> bool:
    key = ""
    try:
        info = inspect_position(game.fen)
        key = (info.key or "").strip()
    except Exception:
        key = ""
    if not key:
        return False

    # Count occurrences: initial key + moves' keys
    count = 0
    if game.initial_position_key and game.initial_position_key == key:
        count += 1
    count += Move.objects.filter(game=game, position_key_after=key).count()
    return count >= 3
