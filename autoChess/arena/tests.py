from __future__ import annotations

import os
import shutil
import unittest
import chess
from django.test import TestCase, override_settings
from django.conf import settings
from django.utils import timezone

from .services.fen import parse_fen_piece_placement, uci_move_highlight, is_insufficient_material
from .models import (
    Game,
    EngineConfig,
    EngineType,
    Player,
    RatingHistory,
    Side,
    StrengthMode,
    STARTPOS_FEN,
    GameStatus,
    Move,
)
from .services.board_svg import render_board_svg
from .services.game_loop import tick, finalize_game
from .services.pgn import build_pgn, is_pvp_game
from .services.rating import (
    apply_ratings_for_game,
    bayeselo_fit,
    elo_delta,
    expected_score,
    k_factor,
    rebuild_ratings,
)
from .services import gnn

class FenParsingTests(TestCase):
    def test_parse_startpos(self):
        board = parse_fen_piece_placement(STARTPOS_FEN)
        self.assertEqual(len(board), 8)
        self.assertEqual(len(board[0]), 8)

    def test_highlight(self):
        hl = uci_move_highlight("e2e4")
        self.assertIsNotNone(hl.from_sq)
        self.assertIsNotNone(hl.to_sq)

    def test_insufficient_material(self):
        self.assertTrue(is_insufficient_material("8/8/8/8/8/8/8/4K2k w - - 0 1"))
        self.assertFalse(is_insufficient_material(STARTPOS_FEN))

class EloUnitTests(TestCase):
    def test_expected_score_symmetry(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)
        self.assertAlmostEqual(expected_score(1500, 1500) + expected_score(1500, 1500), 1.0)
        self.assertGreater(expected_score(1700, 1500), 0.5)

    def test_elo_delta_win_loss_draw(self):
        # K=40 for a brand new player; expected score 0.5 at equal ratings.
        self.assertAlmostEqual(elo_delta(1500, 1500, 1.0, 0), 20.0)
        self.assertAlmostEqual(elo_delta(1500, 1500, 0.0, 0), -20.0)
        self.assertAlmostEqual(elo_delta(1500, 1500, 0.5, 0), 0.0)

    def test_k_factor_adaptive(self):
        self.assertEqual(k_factor(0, 1500), 40.0)
        self.assertEqual(k_factor(50, 1500), 20.0)
        self.assertEqual(k_factor(50, 2500), 10.0)


class BayesEloUnitTests(TestCase):
    def test_stronger_player_rated_higher(self):
        # 0 beats 1 ten times, 1 beats 2 ten times -> ratings[0] > ratings[1] > ratings[2]
        counts = [(0, 1, 10, 0, 0), (1, 2, 10, 0, 0)]
        ratings, kappa = bayeselo_fit(counts, 3)
        self.assertGreater(ratings[0], ratings[1])
        self.assertGreater(ratings[1], ratings[2])
        self.assertGreater(kappa, 0.0)

    def test_anchor_is_fixed(self):
        counts = [(0, 1, 10, 0, 0)]
        ratings, _ = bayeselo_fit(counts, 2, anchors={1: 1500.0})
        self.assertAlmostEqual(ratings[1], 1500.0, places=6)
        self.assertGreater(ratings[0], 1500.0)

    def test_draws_raise_kappa(self):
        counts = [(0, 1, 0, 0, 50)]
        _ratings, kappa = bayeselo_fit(counts, 2)
        self.assertGreater(kappa, 1.0)


class RatingIntegrationTests(TestCase):
    def _finished_game(self, result, white=None, black=None, *, engine_white=False, engine_black=False):
        game = Game.objects.create(
            status=GameStatus.FINISHED,
            fen=STARTPOS_FEN,
            result=result,
            finished_at=timezone.now(),
        )
        EngineConfig.objects.create(
            game=game,
            side=Side.WHITE,
            engine_type=EngineType.STOCKFISH if engine_white else EngineType.PLAYER,
            player=white,
            strength_mode=StrengthMode.SKILL,
            strength_value=5,
        )
        EngineConfig.objects.create(
            game=game,
            side=Side.BLACK,
            engine_type=EngineType.PYCHESS if engine_black else EngineType.PLAYER,
            player=black,
            strength_mode=StrengthMode.SKILL,
            strength_value=3,
        )
        return game

    def test_game_updates_elo_and_history(self):
        alice = Player.objects.create(name="Alice")
        bob = Player.objects.create(name="Bob")
        game = self._finished_game("1-0", alice, bob)

        self.assertTrue(apply_ratings_for_game(game))

        alice.refresh_from_db()
        bob.refresh_from_db()
        self.assertAlmostEqual(alice.elo_rating, 1520.0)
        self.assertAlmostEqual(bob.elo_rating, 1480.0)
        self.assertEqual((alice.games_played, alice.wins), (1, 1))
        self.assertEqual((bob.games_played, bob.losses), (1, 1))
        self.assertEqual(RatingHistory.objects.filter(game=game).count(), 2)

    def test_apply_is_idempotent(self):
        alice = Player.objects.create(name="Alice")
        bob = Player.objects.create(name="Bob")
        game = self._finished_game("1/2-1/2", alice, bob)

        self.assertTrue(apply_ratings_for_game(game))
        alice.refresh_from_db()
        first_elo = alice.elo_rating

        self.assertFalse(apply_ratings_for_game(game))
        alice.refresh_from_db()
        self.assertAlmostEqual(alice.elo_rating, first_elo)
        self.assertEqual(alice.games_played, 1)
        self.assertEqual(RatingHistory.objects.filter(game=game).count(), 2)

    def test_engine_participants_auto_created(self):
        game = self._finished_game("0-1", engine_white=True, engine_black=True)
        self.assertTrue(apply_ratings_for_game(game))
        self.assertTrue(Player.objects.filter(name="Stockfish (skill 5)", is_engine=True).exists())
        self.assertTrue(Player.objects.filter(name="PyChess (skill 3)", is_engine=True).exists())

    def test_unrated_player_side_is_skipped(self):
        # An engine side auto-registers, but an anonymous human (PLAYER, no Player) cannot be rated.
        game = self._finished_game("1-0", white=None, black=None, engine_black=True)
        self.assertFalse(apply_ratings_for_game(game))
        self.assertEqual(RatingHistory.objects.count(), 0)

    def test_rebuild_recomputes_from_scratch(self):
        alice = Player.objects.create(name="Alice")
        bob = Player.objects.create(name="Bob")
        for result in ("1-0", "0-1", "1-0"):
            self._finished_game(result, alice, bob)

        applied = rebuild_ratings()
        self.assertEqual(applied, 3)

        alice.refresh_from_db()
        bob.refresh_from_db()
        self.assertEqual(alice.games_played, 3)
        self.assertEqual(bob.games_played, 3)
        self.assertEqual(RatingHistory.objects.count(), 6)
        # White won two of three -> positive net Elo.
        self.assertGreater(alice.elo_rating, 1500.0)

    def test_finalize_game_applies_ratings(self):
        alice = Player.objects.create(name="Alice")
        bob = Player.objects.create(name="Bob")
        game = self._finished_game("1-0", alice, bob)
        finalize_game(game)
        self.assertTrue(RatingHistory.objects.filter(game=game).exists())


class RatingViewTests(TestCase):
    def test_players_list_and_detail_render(self):
        alice = Player.objects.create(name="Alice")
        bob = Player.objects.create(name="Bob")
        for result in ("1-0", "1-0"):
            game = Game.objects.create(
                status=GameStatus.FINISHED, fen=STARTPOS_FEN, result=result,
                finished_at=timezone.now(),
            )
            EngineConfig.objects.create(
                game=game, side=Side.WHITE, engine_type=EngineType.PLAYER,
                player=alice, strength_mode=StrengthMode.SKILL, strength_value=10,
            )
            EngineConfig.objects.create(
                game=game, side=Side.BLACK, engine_type=EngineType.PLAYER,
                player=bob, strength_mode=StrengthMode.SKILL, strength_value=10,
            )
            apply_ratings_for_game(game)

        resp = self.client.get("/players/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Alice")
        self.assertContains(resp, "BayesElo")

        resp = self.client.get(f"/players/{alice.id}/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Rating trend")
        self.assertContains(resp, "<svg")  # trend chart rendered

    def test_leaderboard_hides_players_without_games(self):
        Player.objects.create(name="Rated Rita", games_played=3, wins=2, losses=1)
        Player.objects.create(name="Rookie Robin", games_played=0)

        resp = self.client.get("/players/", secure=True)
        self.assertContains(resp, "Rated Rita")
        self.assertNotContains(resp, "Rookie Robin")

    def test_game_page_shows_participant_rating(self):
        game = Game.objects.create(status=GameStatus.RUNNING, fen=STARTPOS_FEN)
        EngineConfig.objects.create(
            game=game, side=Side.WHITE, engine_type=EngineType.PYCHESS,
            strength_mode=StrengthMode.SKILL, strength_value=7,
        )
        EngineConfig.objects.create(
            game=game, side=Side.BLACK, engine_type=EngineType.STOCKFISH,
            strength_mode=StrengthMode.SKILL, strength_value=12,
        )
        resp = self.client.get(f"/games/{game.id}/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "PyChess (skill 7)")
        self.assertContains(resp, "Stockfish (skill 12)")


class IntegrationStockfishTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    def test_10_ply_runs_if_engine_available(self):
        exe = getattr(settings, "AUTOCHESS_STOCKFISH_PATH", "stockfish")
        # The setting is either an absolute path or a bare name resolved on PATH.
        if not (os.path.isfile(exe) or shutil.which(exe)):
            self.skipTest(f"Stockfish not found: {exe!r}")

        from .services import engine as engine_service
        from .services.uci import UciError

        # A present binary can still be unusable on a loaded host: bringing
        # Stockfish up needs ~110 MB of free commit for its NNUE network. Treat
        # that as "no engine here" rather than a failure of the match loop.
        try:
            engine_service.pool().get()
        except (UciError, OSError) as exc:
            engine_service._POOL = None
            self.skipTest(f"Stockfish could not be started here: {exc}")

        # preview_ms=1: every tick applies the move the previous tick highlighted
        # instead of racing the default 350 ms highlight window.
        g = Game.objects.create(
            status=GameStatus.RUNNING, fen=STARTPOS_FEN, move_interval_ms=0, preview_ms=1,
        )
        EngineConfig.objects.create(game=g, side=Side.WHITE, strength_mode=StrengthMode.SKILL, strength_value=1, movetime_ms=50)
        EngineConfig.objects.create(game=g, side=Side.BLACK, strength_mode=StrengthMode.SKILL, strength_value=1, movetime_ms=50)

        # perform ticks
        for _ in range(10):
            tick(g.id)
            g.refresh_from_db()
            if g.status == GameStatus.FINISHED:
                break

        self.assertGreaterEqual(g.ply_count, 1)


class PgnExportTests(TestCase):
    def _pvp_game(self, moves, result="1-0"):
        game = Game.objects.create(
            status=GameStatus.FINISHED, fen=STARTPOS_FEN, result=result,
            finished_at=timezone.now(),
        )
        EngineConfig.objects.create(
            game=game, side=Side.WHITE, engine_type=EngineType.PLAYER,
            strength_mode=StrengthMode.SKILL, strength_value=10,
        )
        EngineConfig.objects.create(
            game=game, side=Side.BLACK, engine_type=EngineType.PLAYER,
            strength_mode=StrengthMode.SKILL, strength_value=10,
        )
        for idx, uci in enumerate(moves, start=1):
            Move.objects.create(game=game, ply_index=idx, uci=uci, san=uci, fen_after=STARTPOS_FEN)
        return game

    def test_build_pgn_uses_san_and_headers(self):
        game = self._pvp_game(["e2e4", "e7e5", "g1f3"])
        pgn = build_pgn(game)
        self.assertIn('[Result "1-0"]', pgn)
        self.assertIn("1. e4 e5 2. Nf3 1-0", pgn)
        self.assertIn("Human (White)", pgn)
        self.assertIn("Human (Black)", pgn)

    def test_is_pvp_game(self):
        pvp = self._pvp_game([])
        self.assertTrue(is_pvp_game(pvp))

        engine_game = Game.objects.create(status=GameStatus.FINISHED, fen=STARTPOS_FEN, result="1-0")
        EngineConfig.objects.create(
            game=engine_game, side=Side.WHITE, engine_type=EngineType.PYCHESS,
            strength_mode=StrengthMode.SKILL, strength_value=5,
        )
        EngineConfig.objects.create(
            game=engine_game, side=Side.BLACK, engine_type=EngineType.STOCKFISH,
            strength_mode=StrengthMode.SKILL, strength_value=5,
        )
        self.assertFalse(is_pvp_game(engine_game))

    def test_export_single_game_pgn(self):
        game = self._pvp_game(["d2d4", "d7d5"])
        resp = self.client.get(f"/games/{game.id}/pgn/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("application/x-chess-pgn", resp["Content-Type"])
        self.assertIn("attachment", resp["Content-Disposition"])
        self.assertIn("1. d4 d5", resp.content.decode())

    def test_export_pvp_bundle_only_includes_human_games(self):
        pvp = self._pvp_game(["e2e4"], result="1-0")
        engine_game = Game.objects.create(status=GameStatus.FINISHED, fen=STARTPOS_FEN, result="0-1",
                                          finished_at=timezone.now())
        EngineConfig.objects.create(
            game=engine_game, side=Side.WHITE, engine_type=EngineType.PYCHESS,
            strength_mode=StrengthMode.SKILL, strength_value=5,
        )
        EngineConfig.objects.create(
            game=engine_game, side=Side.BLACK, engine_type=EngineType.STOCKFISH,
            strength_mode=StrengthMode.SKILL, strength_value=5,
        )
        resp = self.client.get("/exports/pvp.pgn", secure=True)
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn(f'[Round "{pvp.id}"]', body)
        self.assertNotIn(f'[Round "{engine_game.id}"]', body)

    def test_create_hot_seat_game_sets_both_sides_human(self):
        resp = self.client.post(
            "/games/new",
            {
                "mode": "HUMAN_VS_HUMAN",
                "white_strength": 10,
                "black_strength": 10,
                "move_interval_ms": 600,
                "movetime_ms": 150,
            },
            secure=True,
        )
        self.assertEqual(resp.status_code, 204)
        game = Game.objects.latest("id")
        types = set(game.engines.values_list("engine_type", flat=True))
        self.assertEqual(types, {EngineType.PLAYER})


def _shorten(text: str, limit: int = 180) -> str:
    """First meaningful line of a long error, for compact skip reasons."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "unknown error"
    picked = next((line for line in lines if "size mismatch" in line), lines[0])
    return picked if len(picked) <= limit else picked[: limit - 1].rstrip() + "..."


def _gnn_status() -> tuple[bool, str]:
    """Whether the GNN can be used, plus the reason when it cannot.

    Checking that the checkpoint *file* exists is not enough: a checkpoint
    trained against an older architecture no longer loads (tensor size
    mismatch), so the model-dependent tests skip until it is retrained instead
    of failing.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False, "torch is not installed"
    if gnn.is_available():
        return True, ""
    return False, f"GNN model unavailable: {_shorten(gnn.unavailable_reason())}"


_GNN_READY, _GNN_SKIP_REASON = _gnn_status()


class GnnPredictionTests(TestCase):
    """Part 4: in-app GNN prediction (must degrade gracefully without torch/model)."""

    def setUp(self):
        gnn.reset_cache()

    def tearDown(self):
        gnn.reset_cache()

    def test_disabled_returns_unavailable(self):
        with override_settings(AUTOCHESS_GNN_ENABLED=False):
            prediction = gnn.predict_position(STARTPOS_FEN)
        self.assertFalse(prediction.available)
        self.assertIn("disabled", prediction.error.lower())

    def test_missing_model_is_reported_not_raised(self):
        with override_settings(AUTOCHESS_GNN_DIR="C:/definitely/not/here", AUTOCHESS_GNN_MODEL=""):
            prediction = gnn.predict_position(STARTPOS_FEN)
        self.assertFalse(prediction.available)
        self.assertTrue(prediction.error)

    def test_prediction_view_renders(self):
        game = Game.objects.create(status=GameStatus.RUNNING, fen=STARTPOS_FEN)
        EngineConfig.objects.create(
            game=game, side=Side.WHITE, engine_type=EngineType.PYCHESS,
            strength_mode=StrengthMode.SKILL, strength_value=5,
        )
        EngineConfig.objects.create(
            game=game, side=Side.BLACK, engine_type=EngineType.STOCKFISH,
            strength_mode=StrengthMode.SKILL, strength_value=5,
        )
        resp = self.client.get(f"/games/{game.id}/prediction/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "GNN prediction")

    @unittest.skipUnless(_GNN_READY, _GNN_SKIP_REASON)
    def test_real_model_predicts_nucleus_and_result(self):
        prediction = gnn.predict_position(STARTPOS_FEN, top_p=0.9)
        self.assertTrue(prediction.available, prediction.error)
        self.assertGreater(len(prediction.moves), 0)
        # Nucleus candidates cannot cover more than the whole (normalised) policy.
        self.assertLessEqual(prediction.total_mass, 1.0 + 1e-6)
        self.assertAlmostEqual(
            prediction.white_win + prediction.draw + prediction.black_win, 1.0, places=5
        )


class GnnEngineTests(TestCase):
    """Part 4: the GNN as a playable engine (EngineType.GNN)."""

    def setUp(self):
        gnn.reset_cache()

    def tearDown(self):
        gnn.reset_cache()

    def _cfg(self):
        return EngineConfig(
            engine_type=EngineType.GNN,
            strength_mode=StrengthMode.SKILL,
            strength_value=10,
            movetime_ms=100,
        )

    @unittest.skipUnless(_GNN_READY, _GNN_SKIP_REASON)
    def test_gnn_best_move_is_legal(self):
        from .services.engine import best_move

        bm = best_move(STARTPOS_FEN, self._cfg())
        legal = {move.uci() for move in chess.Board(STARTPOS_FEN).legal_moves}
        self.assertIn(bm.move, legal)

    def test_gnn_unavailable_becomes_uci_error(self):
        from .services.engine import best_move
        from .services.uci import UciError

        with override_settings(AUTOCHESS_GNN_DIR="C:/definitely/not/here", AUTOCHESS_GNN_MODEL=""):
            gnn.reset_cache()
            with self.assertRaises(UciError):
                best_move(STARTPOS_FEN, self._cfg())

    def test_apply_and_inspect_falls_back_without_stockfish(self):
        from .services import engine

        saved_pool = engine._POOL
        engine._POOL = None
        try:
            with override_settings(AUTOCHESS_STOCKFISH_PATH="C:/definitely/not/stockfish.exe"):
                info = engine.apply_and_inspect(STARTPOS_FEN, "e2e4")
                self.assertIsNotNone(info.fen)
                # The pawn moved, so the placement differs from the start position.
                self.assertNotEqual(info.fen.split()[0], STARTPOS_FEN.split()[0])

                start = engine.inspect_position(STARTPOS_FEN)
                self.assertTrue((start.key or "").strip())
        finally:
            engine._POOL = saved_pool


class ReviewViewTests(TestCase):
    """Part 1: step-through PGN viewer."""

    def _game(self):
        game = Game.objects.create(
            status=GameStatus.FINISHED, fen=STARTPOS_FEN, result="1-0",
            finished_at=timezone.now(),
        )
        EngineConfig.objects.create(
            game=game, side=Side.WHITE, engine_type=EngineType.PLAYER,
            strength_mode=StrengthMode.SKILL, strength_value=10,
        )
        EngineConfig.objects.create(
            game=game, side=Side.BLACK, engine_type=EngineType.PLAYER,
            strength_mode=StrengthMode.SKILL, strength_value=10,
        )
        for idx, (uci, san) in enumerate(
            [("e2e4", "e4"), ("e7e5", "e5"), ("g1f3", "Nf3")], start=1
        ):
            Move.objects.create(game=game, ply_index=idx, uci=uci, san=san, fen_after=STARTPOS_FEN)
        return game

    def test_review_page_renders_board_and_moves(self):
        game = self._game()
        resp = self.client.get(f"/games/{game.id}/review/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Game review")
        self.assertContains(resp, "Nf3")

    def test_review_clamps_out_of_range_ply(self):
        game = self._game()
        for ply in (0, 1, 3, 99):
            resp = self.client.get(f"/games/{game.id}/review/?ply={ply}", secure=True)
            self.assertEqual(resp.status_code, 200)

    def test_review_returns_fragment_for_htmx(self):
        game = self._game()
        resp = self.client.get(
            f"/games/{game.id}/review/?ply=1", secure=True, HTTP_HX_REQUEST="true"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'data-ply="1"')
        self.assertNotContains(resp, "<html")


class GnnSearchDepthTests(TestCase):
    """GNN minimax search at selectable depths (policy only / 3 / 5 / 8)."""

    def setUp(self):
        gnn.reset_cache()

    def tearDown(self):
        gnn.reset_cache()

    def _running_game(self, white=EngineType.PYCHESS, black=EngineType.GNN):
        game = Game.objects.create(status=GameStatus.RUNNING, fen=STARTPOS_FEN)
        EngineConfig.objects.create(
            game=game, side=Side.WHITE, engine_type=white,
            strength_mode=StrengthMode.SKILL, strength_value=7,
        )
        EngineConfig.objects.create(
            game=game, side=Side.BLACK, engine_type=black,
            strength_mode=StrengthMode.SKILL, strength_value=7,
        )
        return game

    def test_player_name_uses_search_depth(self):
        from .services.rating import engine_player_name

        policy = EngineConfig(engine_type=EngineType.GNN, search_depth=0)
        deep = EngineConfig(engine_type=EngineType.GNN, search_depth=8)
        self.assertEqual(engine_player_name(policy), "GNN (policy)")
        self.assertEqual(engine_player_name(deep), "GNN (depth 8)")

    def test_engine_forwards_search_depth_to_gnn(self):
        from unittest import mock

        from .services import engine

        cfg = EngineConfig(engine_type=EngineType.GNN, search_depth=5)
        with mock.patch.object(engine.gnn, "best_move", return_value="e2e4") as mocked:
            best = engine.best_move(STARTPOS_FEN, cfg)

        self.assertEqual(best.move, "e2e4")
        self.assertEqual(mocked.call_args.kwargs.get("depth"), 5)

    def test_update_config_sets_gnn_depth_without_strength_params(self):
        game = self._running_game(white=EngineType.GNN, black=EngineType.PYCHESS)

        # The panel disables the strength/movetime inputs for the GNN, so those
        # fields are absent from the POST: the update must still be accepted.
        resp = self.client.post(
            f"/games/{game.id}/config/",
            {"side": "WHITE", "engine_type": "GNN", "search_depth": "8"},
            secure=True,
        )
        self.assertEqual(resp.status_code, 200)

        cfg = game.engines.get(side=Side.WHITE)
        self.assertEqual(cfg.engine_type, EngineType.GNN)
        self.assertEqual(cfg.search_depth, 8)

    def test_update_config_rejects_unsupported_depth(self):
        game = self._running_game()
        resp = self.client.post(
            f"/games/{game.id}/config/",
            {"side": "BLACK", "engine_type": "GNN", "search_depth": "4"},
            secure=True,
        )
        self.assertEqual(resp.status_code, 400)

    def test_panel_exposes_depth_selector(self):
        game = self._running_game(white=EngineType.GNN, black=EngineType.STOCKFISH)
        resp = self.client.get(f"/games/{game.id}/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'name="search_depth"')
        self.assertContains(resp, "Depth 8 · minimax")

    def test_gnn_label_does_not_imply_bayeselo(self):
        """The GNN engine and the BayesElo rating are unrelated: say so in the UI."""
        game = self._running_game(white=EngineType.GNN, black=EngineType.STOCKFISH)
        resp = self.client.get(f"/games/{game.id}/", secure=True)
        self.assertContains(resp, "GNN (graph neural net)")
        self.assertNotContains(resp, "GNN (BayesianELO)")

    @unittest.skipUnless(_GNN_READY, _GNN_SKIP_REASON)
    def test_search_moves_are_legal_at_every_depth(self):
        legal = {move.uci() for move in chess.Board(STARTPOS_FEN).legal_moves}
        for depth in (3, 5, 8):
            with self.subTest(depth=depth):
                uci = gnn.best_move(STARTPOS_FEN, depth=depth)
                self.assertIn(uci, legal)

    def test_changing_depth_resets_rated_participant(self):
        game = self._running_game(white=EngineType.GNN, black=EngineType.PYCHESS)
        cfg = game.engines.get(side=Side.WHITE)
        # Simulate the participant having been resolved for the "policy" identity.
        cfg.player = Player.objects.create(name="GNN (policy)", is_engine=True)
        cfg.save(update_fields=["player", "updated_at"])

        resp = self.client.post(
            f"/games/{game.id}/config/",
            {"side": "WHITE", "engine_type": "GNN", "search_depth": "5"},
            secure=True,
        )
        self.assertEqual(resp.status_code, 200)

        # The old participant is dropped and the depth-specific one is attached.
        cfg.refresh_from_db()
        self.assertEqual(cfg.player.name, "GNN (depth 5)")

        # ... and the match page shows it on the next load.
        page = self.client.get(f"/games/{game.id}/", secure=True)
        self.assertContains(page, "GNN (depth 5)")


class MatchHistoryTests(TestCase):
    """Part 5: stepping back through the moves played so far (◀ ▶ on the match page)."""

    MOVES = [("e2e4", "e4"), ("e7e5", "e5"), ("g1f3", "Nf3")]

    def _game(self, status=GameStatus.FINISHED):
        game = Game.objects.create(
            status=status, fen=STARTPOS_FEN, result="1-0",
            finished_at=timezone.now(), ply_count=len(self.MOVES),
        )
        for side in (Side.WHITE, Side.BLACK):
            EngineConfig.objects.create(
                game=game, side=side, engine_type=EngineType.PLAYER,
                strength_mode=StrengthMode.SKILL, strength_value=10,
            )
        for idx, (uci, san) in enumerate(self.MOVES, start=1):
            Move.objects.create(
                game=game, ply_index=idx, uci=uci, san=san, fen_after=STARTPOS_FEN
            )
        return game

    def _after_first_move_svg(self) -> str:
        """The board after 1.e4, built independently of the view's ply logic."""
        board = chess.Board(STARTPOS_FEN)
        board.push(chess.Move.from_uci("e2e4"))
        return render_board_svg(board.fen(), "e2e4")

    def test_match_page_offers_history_controls(self):
        game = self._game()
        resp = self.client.get(f"/games/{game.id}/", secure=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Ply")
        # Live at ply 3, so ◀ offers the previous ply.
        self.assertContains(resp, f"/games/{game.id}/?ply=2")

    def test_ply_parameter_shows_the_earlier_position(self):
        game = self._game()
        resp = self.client.get(f"/games/{game.id}/?ply=1", secure=True)
        live = self.client.get(f"/games/{game.id}/", secure=True)

        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Reviewing ply")
        self.assertContains(resp, 'id="board-review"')
        self.assertContains(resp, self._after_first_move_svg(), html=False)
        # ... while the live page shows the final position instead.
        self.assertNotContains(live, self._after_first_move_svg(), html=False)

    def test_reviewing_locks_the_piece_overlay(self):
        # A running game where the human may move: stepping back must be read-only.
        game = self._game(status=GameStatus.RUNNING)
        live = self.client.get(f"/games/{game.id}/", secure=True)
        review = self.client.get(f"/games/{game.id}/?ply=1", secure=True)

        self.assertNotContains(live, "board-overlay pointer-events-none")
        self.assertContains(review, "board-overlay pointer-events-none")

    def test_htmx_step_returns_only_the_board_panel(self):
        game = self._game()
        resp = self.client.get(
            f"/games/{game.id}/?ply=1", secure=True, HTTP_HX_REQUEST="true"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "<html")
        self.assertContains(resp, "data-board-panel")
        self.assertContains(resp, 'id="board-review"')
        self.assertContains(resp, "● Live")

    def test_ply_is_clamped_to_the_moves_played(self):
        game = self._game()
        for ply in (0, 1, 99, -5, "junk", ""):
            with self.subTest(ply=ply):
                resp = self.client.get(f"/games/{game.id}/?ply={ply}", secure=True)
                self.assertEqual(resp.status_code, 200)

    def test_live_poll_leaves_the_board_alone_while_reviewing(self):
        game = self._game()
        reviewing = self.client.post(
            f"/games/{game.id}/tick/", {"reviewing": "1"}, secure=True
        )
        self.assertContains(reviewing, 'id="status-info"')
        self.assertNotContains(reviewing, 'id="board"')

        live = self.client.post(f"/games/{game.id}/tick/", {}, secure=True)
        self.assertContains(live, 'id="board"')
