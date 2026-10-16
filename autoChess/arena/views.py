from __future__ import annotations

from datetime import timedelta

import chess

from django.conf import settings
from django.db import transaction
from django.db.utils import OperationalError
from django.utils import timezone
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from .forms import NewGameForm, UpdateConfigForm, UpdateSpeedForm
from .models import Game, EngineConfig, Player, Side, StrengthMode, GameStatus, EngineType, STARTPOS_FEN, Move, TerminationReason
from .services.board_svg import render_board_svg
from .services.chart import rating_chart_svg
from .services.fen import halfmove_clock_from_fen, is_insufficient_material
from .services.game_loop import tick, ensure_engine_rows, finalize_game
from .services.pgn import build_pgn, is_pvp_game
from .services.rating import engine_player_name, ensure_participants
from .services.engine import inspect_position
from .services.gnn import predict_position


def _player_can_move(game: Game) -> bool:
    # Allow clicking only if the game is running, there is no pending engine preview,
    # and the side to move is PLAYER (human).
    if game.status != GameStatus.RUNNING:
        return False
    if (game.pending_move_uci or "").strip():
        return False
    try:
        stm = (game.fen.split()[1] if game.fen else game.side_to_move).strip()
    except Exception:
        stm = game.side_to_move
    side = Side.WHITE if stm == "w" else Side.BLACK
    try:
        cfg = game.engines.get(side=side)
    except Exception:
        return False
    return cfg.engine_type == EngineType.PLAYER


def _is_valid_square(s: str) -> bool:
    s = (s or "").strip().lower()
    return len(s) == 2 and s[0] in "abcdefgh" and s[1] in "12345678"


def dashboard(request: HttpRequest) -> HttpResponse:
    """Dashboard: new match + recent results."""
    form = NewGameForm()
    recent = Game.objects.filter(status=GameStatus.FINISHED).order_by("-finished_at")[:20]
    return render(request, "arena/dashboard.html", {"form": form, "recent": recent})


@require_http_methods(["POST"])
def create_game(request: HttpRequest) -> HttpResponse:
    form = NewGameForm(request.POST)
    if not form.is_valid():
        return HttpResponseBadRequest(
            render(request, "arena/fragments/new_game_form.html", {"form": form}).content
        )

    with transaction.atomic():
        game = Game.objects.create(
            status=GameStatus.CONFIGURED,
            fen=STARTPOS_FEN,
            side_to_move="w",
            move_interval_ms=form.cleaned_data["move_interval_ms"],
            preview_ms=int(form.cleaned_data.get("preview_ms") or getattr(settings, "AUTOCHESS_PREVIEW_MS", 350)),
            next_action_at=None,
        )

        # The selected game mode decides which sides are human (local hot-seat).
        mode = form.cleaned_data.get("mode", NewGameForm.MODE_ENGINE)
        white_type = EngineType.PYCHESS
        black_type = EngineType.STOCKFISH
        if mode == NewGameForm.MODE_HUMAN_WHITE:
            white_type = EngineType.PLAYER
        elif mode == NewGameForm.MODE_HUMAN_VS_HUMAN:
            white_type = EngineType.PLAYER
            black_type = EngineType.PLAYER

        EngineConfig.objects.create(
            game=game,
            side=Side.WHITE,
            engine_type=white_type,
            strength_mode=StrengthMode.SKILL,
            strength_value=form.cleaned_data["white_strength"],
            movetime_ms=form.cleaned_data["movetime_ms"],
        )
        EngineConfig.objects.create(
            game=game,
            side=Side.BLACK,
            engine_type=black_type,
            strength_mode=StrengthMode.SKILL,
            strength_value=form.cleaned_data["black_strength"],
            movetime_ms=form.cleaned_data["movetime_ms"],
        )
        # Initialize repetition key (Stockfish 'Key:')
        try:
            info = inspect_position(game.fen)
            game.initial_position_key = (info.key or "")
            game.save(update_fields=["initial_position_key"])
        except Exception:
            pass

        # Register rated participants for engine sides (so ratings can be shown).
        ensure_participants(game)

    resp = HttpResponse(status=204)
    resp["HX-Redirect"] = reverse("arena:game", kwargs={"game_id": game.id})
    return resp


def game_view(request: HttpRequest, game_id: int) -> HttpResponse:
    game = get_object_or_404(Game, pk=game_id)
    ensure_engine_rows(game)
    ensure_participants(game)
    board_svg = render_board_svg(game.fen, game.pending_move_uci or game.last_move_uci)
    ctx = {
        "game": game,
        "board_svg": board_svg,
        "player_can_move": _player_can_move(game),
        "white": game.engines.get(side=Side.WHITE),
        "black": game.engines.get(side=Side.BLACK),
                "config_form": UpdateConfigForm(
            initial={
                "strength_mode": StrengthMode.SKILL,
                "strength_value": 10,
                "movetime_ms": 150,
                "side": "WHITE",
            }
        ),
            }
    return render(request, "arena/game.html", ctx)


@require_http_methods(["POST"])
def control(request: HttpRequest, game_id: int, action: str) -> HttpResponse:
    _ = get_object_or_404(Game, pk=game_id)
    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        ensure_engine_rows(game)

        if action == "start":
            game.mark_started()
            game.save()
        elif action == "pause":
            game.mark_paused()
            game.save()
        elif action == "reset":
            new = Game.objects.create(
                status=GameStatus.CONFIGURED,
                fen=STARTPOS_FEN,
                side_to_move="w",
                move_interval_ms=game.move_interval_ms,
                preview_ms=getattr(game, "preview_ms", 350),
            )
            white = game.engines.get(side=Side.WHITE)
            black = game.engines.get(side=Side.BLACK)
            EngineConfig.objects.create(
                game=new,
                side=Side.WHITE,
                engine_type=white.engine_type,
                strength_mode=white.strength_mode,
                strength_value=white.strength_value,
                movetime_ms=white.movetime_ms,
            )
            EngineConfig.objects.create(
                game=new,
                side=Side.BLACK,
                engine_type=black.engine_type,
                strength_mode=black.strength_mode,
                strength_value=black.strength_value,
                movetime_ms=black.movetime_ms,
            )
            resp = HttpResponse(status=204)
            resp["HX-Redirect"] = reverse("arena:game", kwargs={"game_id": new.id})
            return resp
        else:
            return HttpResponseBadRequest("Ismeretlen action.")

    return _render_tick_oob(request, game)


@require_http_methods(["POST"])
def update_config(request: HttpRequest, game_id: int) -> HttpResponse:
    _ = get_object_or_404(Game, pk=game_id)
    form = UpdateConfigForm(request.POST)
    if not form.is_valid():
        return HttpResponseBadRequest("Invalid configuration.")

    side = form.cleaned_data["side"]
    new_type = form.cleaned_data["engine_type"]

    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        ensure_engine_rows(game)
        cfg = game.engines.get(side=side)

        # Rated identity of an engine side is derived from its type and its
        # strength (or, for the GNN, its search depth). Keep the previous name so
        # we can tell whether the participant must be re-resolved below.
        previous_name = (
            engine_player_name(cfg) if cfg.engine_type != EngineType.PLAYER else ""
        )

        cfg.engine_type = new_type

        if new_type == EngineType.GNN:
            # The GNN is configured by its minimax search depth; the UCI
            # strength mode/value and thinking time do not apply to it.
            cfg.search_depth = form.cleaned_data.get("search_depth") or 0
        elif new_type != EngineType.PLAYER:
            # For PLAYER (human) the strength/movetime fields are ignored.
            cfg.strength_mode = form.cleaned_data["strength_mode"]
            cfg.strength_value = form.cleaned_data["strength_value"]
            cfg.movetime_ms = form.cleaned_data["movetime_ms"]

        # Changing the engine identity (e.g. GNN policy -> depth 5) must not keep
        # crediting the old rated player, so drop the cached participant. It is
        # resolved again from the new name (and never touched for finished games).
        if (
            new_type != EngineType.PLAYER
            and game.status != GameStatus.FINISHED
            and engine_player_name(cfg) != previous_name
        ):
            cfg.player = None

        cfg.save()

        # Re-resolve rated participants so the panel shows the new identity
        # (e.g. "GNN (depth 8)") immediately after the change.
        ensure_participants(game)

        # If we switch the side to move to PLAYER, clear any pending engine preview.
        try:
            stm = (game.fen.split()[1] if game.fen else game.side_to_move).strip()
        except Exception:
            stm = game.side_to_move
        cur_side = Side.WHITE if stm == "w" else Side.BLACK
        if new_type == EngineType.PLAYER and cur_side == side and (game.pending_move_uci or "").strip():
            game.pending_move_uci = ""
            game.pending_move_set_at = None
            game.save(update_fields=["pending_move_uci", "pending_move_set_at"])

    return _render_full_oob(request, game)



@require_http_methods(["POST"])
def update_speed(request: HttpRequest, game_id: int) -> HttpResponse:
    _ = get_object_or_404(Game, pk=game_id)
    form = UpdateSpeedForm(request.POST)
    if not form.is_valid():
        return HttpResponseBadRequest("Invalid speed.")

    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        game.move_interval_ms = form.cleaned_data["move_interval_ms"]
        game.preview_ms = form.cleaned_data["preview_ms"]
        game.save(update_fields=["move_interval_ms", "preview_ms", "updated_at"])

    return _render_full_oob(request, game)


@require_http_methods(["POST"])
def click_square_view(request: HttpRequest, game_id: int) -> HttpResponse:
    """Handle a player click: selection + move validation and execution."""
    square = (request.POST.get("square") or "").strip().lower()
    if not _is_valid_square(square):
        return HttpResponseBadRequest("Invalid square.")

    try:
        with transaction.atomic():
            game = Game.objects.select_for_update().get(pk=game_id)
            ensure_engine_rows(game)

            # Only allow player input while the game is running.
            if game.status != GameStatus.RUNNING:
                if game.ui_selected_from or game.ui_selected_to:
                    game.ui_selected_from = ""
                    game.ui_selected_to = ""
                    game.save(update_fields=["ui_selected_from", "ui_selected_to"])
                return _render_tick_oob(request, game)

            # Do not allow clicking while an engine preview is pending.
            if (game.pending_move_uci or "").strip():
                if game.ui_selected_from or game.ui_selected_to:
                    game.ui_selected_from = ""
                    game.ui_selected_to = ""
                    game.save(update_fields=["ui_selected_from", "ui_selected_to"])
                return _render_tick_oob(request, game)

            # Side to move and its config
            stm = (game.fen.split()[1] if game.fen else game.side_to_move).strip()
            side = Side.WHITE if stm == "w" else Side.BLACK
            cfg = game.engines.get(side=side)
            if cfg.engine_type != EngineType.PLAYER:
                if game.ui_selected_from or game.ui_selected_to:
                    game.ui_selected_from = ""
                    game.ui_selected_to = ""
                    game.save(update_fields=["ui_selected_from", "ui_selected_to"])
                return _render_tick_oob(request, game)

            board = chess.Board(game.fen)

            # 1) Selection
            if not (game.ui_selected_from or "").strip():
                piece = board.piece_at(chess.parse_square(square))
                if piece is None or piece.color != board.turn:
                    return _render_tick_oob(request, game)

                game.ui_selected_from = square
                game.ui_selected_to = ""
                game.save(update_fields=["ui_selected_from", "ui_selected_to"])
                return _render_tick_oob(request, game)

            # 2) Second click: attempt a move or deselect
            from_sq = (game.ui_selected_from or "").strip().lower()
            if square == from_sq:
                game.ui_selected_from = ""
                game.ui_selected_to = ""
                game.save(update_fields=["ui_selected_from", "ui_selected_to"])
                return _render_tick_oob(request, game)

            # Local target selection (visual only) - the server response is the source of truth.
            game.ui_selected_to = square
            game.save(update_fields=["ui_selected_to"])

            # Build the move (promotion: queen by default)
            uci = f"{from_sq}{square}"
            move = chess.Move.from_uci(uci)
            if board.piece_at(chess.parse_square(from_sq)) and board.piece_at(chess.parse_square(from_sq)).piece_type == chess.PAWN:
                to_rank = int(square[1])
                if (board.turn == chess.WHITE and to_rank == 8) or (board.turn == chess.BLACK and to_rank == 1):
                    move = chess.Move.from_uci(uci + "q")

            if move not in board.legal_moves:
                # Illegal: clear the selection
                game.ui_selected_from = ""
                game.ui_selected_to = ""
                game.save(update_fields=["ui_selected_from", "ui_selected_to"])
                return _render_tick_oob(request, game)

            # Legal: apply + save
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()

            board.push(move)
            fen_after = board.fen()
            try:
                is_check = board.is_check()
            except Exception:
                is_check = False

            # Position key (repetition detection)
            pos_key = ""
            try:
                info = inspect_position(fen_after)
                pos_key = (info.key or "").strip()
            except Exception:
                pos_key = ""

            ply_index = int(game.ply_count) + 1

            Move.objects.create(
                game=game,
                ply_index=ply_index,
                uci=move.uci(),
                san=san,
                fen_after=fen_after,
                position_key_after=pos_key,
                is_check=is_check,
            )

            # Update the game
            game.fen = fen_after
            game.side_to_move = "w" if board.turn == chess.WHITE else "b"
            game.ply_count = ply_index
            game.last_move_uci = move.uci()
            game.last_move_san = san
            game.ui_selected_from = ""
            game.ui_selected_to = ""
            game.pending_move_uci = ""
            game.pending_move_set_at = None

            # Schedule the next engine preview so the whole cycle is ~move_interval.
            remaining_ms = max(0, int(game.move_interval_ms) - int(game.preview_ms or 0))
            game.next_action_at = timezone.now() + timedelta(milliseconds=remaining_ms)
            game.save()

            # Terminal-state checks
            max_plies = int(getattr(settings, "AUTOCHESS_MAX_PLIES", 600))
            if ply_index >= max_plies:
                game.mark_finished(result="1/2-1/2", reason=TerminationReason.MAX_PLIES)
            elif board.is_checkmate():
                winner = "1-0" if side == Side.WHITE else "0-1"
                game.mark_finished(result=winner, reason=TerminationReason.CHECKMATE)
            elif board.is_stalemate():
                game.mark_finished(result="1/2-1/2", reason=TerminationReason.STALEMATE)
            elif pos_key:
                count = 0
                if game.initial_position_key and game.initial_position_key == pos_key:
                    count += 1
                count += Move.objects.filter(game=game, position_key_after=pos_key).count()
                if count >= 3:
                    game.mark_finished(result="1/2-1/2", reason=TerminationReason.REPETITION)

            if game.status != GameStatus.FINISHED:
                try:
                    if halfmove_clock_from_fen(game.fen) >= 100:
                        game.mark_finished(result="1/2-1/2", reason=TerminationReason.FIFTY_MOVE)
                except Exception:
                    pass

            if game.status != GameStatus.FINISHED:
                try:
                    if is_insufficient_material(game.fen):
                        game.mark_finished(result="1/2-1/2", reason=TerminationReason.INSUFFICIENT)
                except Exception:
                    pass

            if game.status == GameStatus.FINISHED:
                game.save(update_fields=["status", "result", "termination_reason", "finished_at"])
                finalize_game(game)
    except OperationalError as e:
        if "database is locked" not in str(e).lower():
            raise

    game = get_object_or_404(Game, pk=game_id)
    return _render_tick_oob(request, game)

@require_http_methods(["POST"])
def tick_view(request: HttpRequest, game_id: int) -> HttpResponse:
    try:
        _ = tick(game_id)
    except OperationalError as e:
        # Under SQLite a concurrent tick + control (e.g. pause) can briefly hit 'database is locked'.
        # In that case just drop the tick; the next poll will refresh anyway.
        if 'database is locked' not in str(e).lower():
            raise
    game = get_object_or_404(Game, pk=game_id)
    return _render_tick_oob(request, game)


def _render_tick_oob(request: HttpRequest, game: Game) -> HttpResponse:
    board_svg = render_board_svg(game.fen, game.pending_move_uci or game.last_move_uci)
    ctx = {
        "game": game,
        "board_svg": board_svg,
        "player_can_move": _player_can_move(game),
        "white": game.engines.get(side=Side.WHITE),
        "black": game.engines.get(side=Side.BLACK),
    }
    return render(request, "arena/fragments/tick_response.html", ctx)

def _render_full_oob(request: HttpRequest, game: Game) -> HttpResponse:
    board_svg = render_board_svg(game.fen, game.pending_move_uci or game.last_move_uci)
    ctx = {
        "game": game,
        "board_svg": board_svg,
        "player_can_move": _player_can_move(game),
        "white": game.engines.get(side=Side.WHITE),
        "black": game.engines.get(side=Side.BLACK),
    }
    return render(request, "arena/fragments/full_response.html", ctx)


@require_http_methods(["GET"]) 
def moves_view(request: HttpRequest, game_id: int) -> HttpResponse:
    game = get_object_or_404(Game, pk=game_id)
    ctx = {"game": game, "moves": game.moves.all()}
    return render(request, "arena/fragments/moves.html", ctx)


@require_http_methods(["GET"])
def prediction_view(request: HttpRequest, game_id: int) -> HttpResponse:
    """Live GNN prediction for the current position (HTMX fragment)."""
    game = get_object_or_404(Game, pk=game_id)

    raw_top_p = request.GET.get("top_p")
    top_p: float | None
    try:
        top_p = float(raw_top_p) if raw_top_p not in (None, "") else None
    except (TypeError, ValueError):
        top_p = None

    prediction = predict_position(game.fen, top_p)
    return render(request, "arena/fragments/prediction.html", {"game": game, "prediction": prediction})


def _fen_at_ply(game: Game, ply: int) -> tuple[str, str]:
    """Return ``(fen, last_move_uci)`` after ``ply`` half-moves (ply=0 = start)."""
    board = chess.Board(STARTPOS_FEN)
    last_uci = ""
    for move_row in game.moves.order_by("ply_index"):
        if move_row.ply_index > ply:
            break
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
        board.push(move)
        last_uci = move.uci()
    return board.fen(), last_uci


@require_http_methods(["GET"])
def review_view(request: HttpRequest, game_id: int) -> HttpResponse:
    """Step-through viewer: the board at any ply, with navigation controls."""
    game = get_object_or_404(Game, pk=game_id)
    moves = list(game.moves.order_by("ply_index"))
    total = len(moves)

    try:
        ply = int(request.GET.get("ply", total))
    except (TypeError, ValueError):
        ply = total
    ply = max(0, min(ply, total))

    fen, last_uci = _fen_at_ply(game, ply)
    ctx = {
        "game": game,
        "moves": moves,
        "ply": ply,
        "total": total,
        "first_ply": max(0, ply - 1) if ply > 0 else 0,
        "board_svg": render_board_svg(fen, last_uci),
        "fen": fen,
    }
    if request.headers.get("HX-Request"):
        return render(request, "arena/fragments/review_board.html", ctx)
    return render(request, "arena/review.html", ctx)


@require_http_methods(["GET"])
def players_view(request: HttpRequest) -> HttpResponse:
    """Leaderboard: rated participants ordered by Elo."""
    players = Player.objects.all()
    return render(request, "arena/players.html", {"players": players})


@require_http_methods(["GET"])
def player_detail_view(request: HttpRequest, player_id: int) -> HttpResponse:
    """Player profile: statistics and a rating trend chart."""
    player = get_object_or_404(Player, pk=player_id)
    history = list(player.rating_history.select_related("game").order_by("created_at", "id"))
    return render(
        request,
        "arena/player_detail.html",
        {
            "player": player,
            "history": history,
            "chart_svg": rating_chart_svg(history),
        },
    )


def _pgn_response(body: str, filename: str) -> HttpResponse:
    resp = HttpResponse(body, content_type="application/x-chess-pgn; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


@require_http_methods(["GET"])
def export_game_pgn(request: HttpRequest, game_id: int) -> HttpResponse:
    """Export a single game as a downloadable PGN file."""
    game = get_object_or_404(Game, pk=game_id)
    return _pgn_response(build_pgn(game), f"autochess-game-{game.id}.pgn")


@require_http_methods(["GET"])
def export_pvp_pgn(request: HttpRequest) -> HttpResponse:
    """Export every finished human-vs-human (hot-seat) match as one PGN file."""
    games = (
        Game.objects.filter(status=GameStatus.FINISHED)
        .prefetch_related("engines", "moves")
        .order_by("finished_at", "id")
    )
    chunks = [build_pgn(g) for g in games if is_pvp_game(g)]
    body = "\n".join(chunk.rstrip() + "\n" for chunk in chunks)
    return _pgn_response(body, "autochess-pvp.pgn")
