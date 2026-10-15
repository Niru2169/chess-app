from __future__ import annotations

"""Rating system: Elo (incremental) and BayesElo (batch fit).

Elo
----
Classic formula with an adaptive K factor::

    expected = 1 / (1 + 10 ** ((opp - rating) / 400))
    new = old + K * (actual - expected)

BayesElo
--------
Bayesian likelihood using the Coulom model. For a pair (i, j):

    denom = rho_i + rho_j + kappa * sqrt(rho_i * rho_j)

    P(i wins)  = rho_i / denom
    P(j wins)  = rho_j / denom
    P(draw)    = kappa * sqrt(rho_i * rho_j) / denom

where ``rho = exp(u)`` and ``u = rating * ln(10) / 400``. ``kappa`` is the draw
parameter. The log-posterior is maximized by gradient ascent (backtracking line
search). Because the objective is concave, this converges to the global maximum.
No scipy/numpy dependency.
"""

import math
from collections import defaultdict

from django.conf import settings
from django.db import transaction
from django.db.models import Q

from ..models import (
    DEFAULT_RATING,
    EngineConfig,
    EngineType,
    Game,
    GameStatus,
    Player,
    RatingHistory,
    Side,
    StrengthMode,
)

ELO_SCALE = 400.0
K_NEW = 40.0          # new players
K_ESTABLISHED = 20.0  # established players
K_MASTER = 10.0       # above 2400
NEW_PLAYER_GAMES = 30
MASTER_RATING = 2400.0

_RESULT_SCORES = {
    "1-0": (1.0, 0.0),
    "0-1": (0.0, 1.0),
    "1/2-1/2": (0.5, 0.5),
}


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #

def expected_score(rating: float, opponent_rating: float) -> float:
    """Expected score (0..1) against the given opponent."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / ELO_SCALE))


def k_factor(games_played: int, rating: float) -> float:
    """Adaptive K factor: higher for new players, lower for outstanding ratings."""
    if games_played < NEW_PLAYER_GAMES:
        return K_NEW
    if rating >= MASTER_RATING:
        return K_MASTER
    return K_ESTABLISHED


def elo_delta(rating: float, opponent_rating: float, actual: float, games_played: int) -> float:
    """The Elo change for a single match."""
    return k_factor(games_played, rating) * (actual - expected_score(rating, opponent_rating))


# --------------------------------------------------------------------------- #
# BayesElo
# --------------------------------------------------------------------------- #

def bayeselo_fit(
    game_counts: list[tuple[int, int, int, int, int]],
    n_players: int,
    *,
    anchors: dict[int, float] | None = None,
    prior_rating: float = DEFAULT_RATING,
    prior_std: float = 300.0,
    kappa_prior_std: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[list[float], float]:
    """BayesElo fit.

    Args:
        game_counts: (i, j, wins_i, wins_j, draws) aggregated results.
        n_players: number of players (indices 0..n-1).
        anchors: {index: known_rating} for pinned players.
        prior_rating: prior mean.
        prior_std: prior standard deviation (Elo).
        kappa_prior_std: prior standard deviation of the draw parameter.

    Returns:
        (ratings, kappa): per-player ratings and the draw parameter.
    """
    anchors = anchors or {}
    scale = math.log(10.0) / ELO_SCALE
    prior_std_u = max(1e-9, prior_std * scale)
    kappa_prior_std = max(1e-9, kappa_prior_std)

    fixed_u = [0.0] * n_players
    for p, rating in anchors.items():
        fixed_u[p] = (rating - prior_rating) * scale

    free = [p for p in range(n_players) if p not in anchors]
    index_of: dict[int, int] = {p: k for k, p in enumerate(free)}

    def unpack(u_free: list[float]) -> list[float]:
        uu = fixed_u[:]
        for k, p in enumerate(free):
            uu[p] = u_free[k]
        return uu

    total_games = sum(wi + wj + d for _, _, wi, wj, d in game_counts)

    def objective(uu: list[float], ln_kappa: float) -> float:
        kappa = math.exp(ln_kappa)
        ll = 0.0
        for i, j, wi, wj, d in game_counts:
            ri, rj = math.exp(uu[i]), math.exp(uu[j])
            s = math.sqrt(ri * rj)
            denom = ri + rj + kappa * s
            ll += wi * uu[i] + wj * uu[j] + d * (ln_kappa + math.log(s))
            ll -= (wi + wj + d) * math.log(denom)
        ll -= sum(uu[p] ** 2 for p in free) / (2.0 * prior_std_u ** 2)
        ll -= ln_kappa ** 2 / (2.0 * kappa_prior_std ** 2)
        return ll

    def gradient(uu: list[float], ln_kappa: float) -> tuple[list[float], float]:
        kappa = math.exp(ln_kappa)
        grad = [0.0] * len(free)
        grad_kappa = 0.0
        for i, j, wi, wj, d in game_counts:
            ri, rj = math.exp(uu[i]), math.exp(uu[j])
            s = math.sqrt(ri * rj)
            denom = ri + rj + kappa * s
            n = wi + wj + d
            gi = wi + d / 2.0 - n * (ri + kappa * s / 2.0) / denom
            gj = wj + d / 2.0 - n * (rj + kappa * s / 2.0) / denom
            grad_kappa += d - n * (kappa * s) / denom
            ki = index_of.get(i)
            if ki is not None:
                grad[ki] += gi
            kj = index_of.get(j)
            if kj is not None:
                grad[kj] += gj
        for k, p in enumerate(free):
            grad[k] -= uu[p] / (prior_std_u ** 2)
        grad_kappa -= ln_kappa / (kappa_prior_std ** 2)
        return grad, grad_kappa

    u_free = [0.0] * len(free)
    ln_kappa = 0.0

    if not game_counts:
        return [prior_rating + u / scale for u in fixed_u], 1.0

    best = objective(unpack(u_free), ln_kappa)
    step = 1.0 / max(1, total_games)

    for _ in range(max_iter):
        grad, grad_kappa = gradient(unpack(u_free), ln_kappa)
        norm = math.sqrt(sum(g * g for g in grad) + grad_kappa * grad_kappa)
        if norm < tol:
            break
        improved = False
        for _ in range(80):
            cand = [u_free[k] + step * grad[k] for k in range(len(free))]
            cand_kappa = ln_kappa + step * grad_kappa
            value = objective(unpack(cand), cand_kappa)
            if value > best + 1e-12:
                u_free, ln_kappa, best = cand, cand_kappa, value
                step *= 1.5
                improved = True
                break
            step *= 0.5
        if not improved:
            break

    uu = unpack(u_free)
    ratings = [prior_rating + uu[p] / scale for p in range(n_players)]
    return ratings, math.exp(ln_kappa)


# --------------------------------------------------------------------------- #
# Participant resolution
# --------------------------------------------------------------------------- #

def engine_player_name(cfg: EngineConfig) -> str:
    """Stable, descriptive name for an engine (based on its strength level)."""
    base = {
        EngineType.STOCKFISH: "Stockfish",
        EngineType.PYCHESS: "PyChess",
    }.get(cfg.engine_type, str(cfg.engine_type).title())
    if cfg.strength_mode == StrengthMode.ELO:
        return f"{base} (elo {cfg.strength_value})"
    return f"{base} (skill {cfg.strength_value})"


def _participant_for_cfg(cfg: EngineConfig) -> Player | None:
    """The rated participant for a side, or None if it cannot be rated.

    - If the EngineConfig has a linked Player, that one is used.
    - A human (PLAYER) side without a Player cannot be rated.
    - For an engine side a Player (is_engine=True) is created automatically
      if ``AUTOCHESS_RATE_ENGINES`` is enabled.
    """
    if cfg.player_id:
        return cfg.player
    if cfg.engine_type == EngineType.PLAYER:
        return None
    if not getattr(settings, "AUTOCHESS_RATE_ENGINES", True):
        return None
    player, _ = Player.objects.get_or_create(
        name=engine_player_name(cfg),
        defaults={"is_engine": True},
    )
    return player


def ensure_participants(game: Game) -> tuple[Player | None, Player | None]:
    """Resolve participants and attach them to the EngineConfigs.

    For engine sides it creates the Player record automatically, so the rating
    can be shown even before the match starts.
    """
    return _resolve_participants(game)


def _resolve_participants(game: Game) -> tuple[Player | None, Player | None]:
    """Resolve the participants of both sides and attach them to the EngineConfigs."""
    cfgs = {c.side: c for c in game.engines.all()}
    cfg_w = cfgs.get(Side.WHITE)
    cfg_b = cfgs.get(Side.BLACK)
    if cfg_w is None or cfg_b is None:
        return None, None

    white = _participant_for_cfg(cfg_w)
    black = _participant_for_cfg(cfg_b)

    if white is not None and cfg_w.player_id != white.pk:
        cfg_w.player = white
        cfg_w.save(update_fields=["player", "updated_at"])
    if black is not None and cfg_b.player_id != black.pk:
        cfg_b.player = black
        cfg_b.save(update_fields=["player", "updated_at"])

    return white, black


def _result_scores(game: Game) -> tuple[float, float]:
    return _RESULT_SCORES.get(game.result, (0.5, 0.5))


def _bump_stats(player: Player, score: float) -> None:
    player.games_played += 1
    if score >= 1.0:
        player.wins += 1
    elif score <= 0.0:
        player.losses += 1
    else:
        player.draws += 1


# --------------------------------------------------------------------------- #
# BayesElo batch refit
# --------------------------------------------------------------------------- #

def _rated_games_upto(upto_game: Game | None) -> list[tuple[Player, Player, float, float]]:
    """Rated finished matches (both sides have a Player), in chronological order up to a prefix."""
    qs = Game.objects.filter(status=GameStatus.FINISHED)
    if upto_game is not None and upto_game.finished_at is not None:
        qs = qs.filter(
            Q(finished_at__lt=upto_game.finished_at)
            | Q(finished_at=upto_game.finished_at, id__lte=upto_game.id)
        )
    qs = qs.order_by("finished_at", "id").prefetch_related("engines__player")

    rated: list[tuple[Player, Player, float, float]] = []
    for game in qs:
        cfgs = {c.side: c for c in game.engines.all()}
        cfg_w = cfgs.get(Side.WHITE)
        cfg_b = cfgs.get(Side.BLACK)
        if not (cfg_w and cfg_b and cfg_w.player_id and cfg_b.player_id):
            continue
        if cfg_w.player_id == cfg_b.player_id:
            continue
        ws, bs = _result_scores(game)
        rated.append((cfg_w.player, cfg_b.player, ws, bs))
    return rated


def _refit_bayeselo(upto_game: Game | None) -> dict[int, float]:
    """Recompute and save BayesElo. Returns the {player_id: rating} mapping."""
    rated = _rated_games_upto(upto_game)
    if not rated:
        return {}

    index: dict[int, int] = {}
    players: list[Player] = []
    for white, black, _, _ in rated:
        for p in (white, black):
            if p.pk not in index:
                index[p.pk] = len(players)
                players.append(p)

    if len(players) < 2:
        return {}

    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for white, black, ws, bs in rated:
        i, j = index[white.pk], index[black.pk]
        if i == j:
            continue
        if i > j:
            i, j, ws, bs = j, i, bs, ws
        if ws >= 1.0:
            counts[(i, j)][0] += 1
        elif bs >= 1.0:
            counts[(i, j)][1] += 1
        else:
            counts[(i, j)][2] += 1

    game_counts = [(i, j, w, l, d) for (i, j), (w, l, d) in counts.items()]
    prior_std = float(getattr(settings, "AUTOCHESS_BAYESELO_PRIOR_STD", 300.0))
    ratings, _kappa = bayeselo_fit(game_counts, len(players), prior_std=prior_std)

    result: dict[int, float] = {}
    for p in players:
        rating = ratings[index[p.pk]]
        result[p.pk] = rating

    # Only write back values that actually changed.
    to_update: list[Player] = []
    for p in players:
        new_rating = result[p.pk]
        if abs(p.bayeselo_rating - new_rating) > 1e-9:
            p.bayeselo_rating = new_rating
            to_update.append(p)
    if to_update:
        Player.objects.bulk_update(to_update, ["bayeselo_rating"])
    return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@transaction.atomic
def apply_ratings_for_game(game: Game) -> bool:
    """Rate a finished match (Elo + BayesElo). Idempotent.

    Returns:
        True if the rating was applied now; False if it cannot be rated or was already applied.
    """
    if game.status != GameStatus.FINISHED:
        return False

    if RatingHistory.objects.filter(game=game).exists():
        return False

    white, black = _resolve_participants(game)
    if white is None or black is None or white.pk == black.pk:
        return False

    ws, bs = _result_scores(game)

    pre_w_elo, pre_b_elo = white.elo_rating, black.elo_rating
    pre_w_bayes, pre_b_bayes = white.bayeselo_rating, black.bayeselo_rating

    d_w = elo_delta(pre_w_elo, pre_b_elo, ws, white.games_played)
    d_b = elo_delta(pre_b_elo, pre_w_elo, bs, black.games_played)

    _bump_stats(white, ws)
    _bump_stats(black, bs)
    white.elo_rating = pre_w_elo + d_w
    black.elo_rating = pre_b_elo + d_b
    white.save(update_fields=[
        "elo_rating", "games_played", "wins", "losses", "draws", "updated_at",
    ])
    black.save(update_fields=[
        "elo_rating", "games_played", "wins", "losses", "draws", "updated_at",
    ])

    _refit_bayeselo(game)

    white.refresh_from_db(fields=["bayeselo_rating"])
    black.refresh_from_db(fields=["bayeselo_rating"])

    RatingHistory.objects.bulk_create([
        RatingHistory(
            player=white,
            game=game,
            elo_rating=white.elo_rating,
            bayeselo_rating=white.bayeselo_rating,
            elo_delta=d_w,
            bayeselo_delta=white.bayeselo_rating - pre_w_bayes,
        ),
        RatingHistory(
            player=black,
            game=game,
            elo_rating=black.elo_rating,
            bayeselo_rating=black.bayeselo_rating,
            elo_delta=d_b,
            bayeselo_delta=black.bayeselo_rating - pre_b_bayes,
        ),
    ])
    return True


@transaction.atomic
def rebuild_ratings() -> int:
    """Recompute every rating from scratch (in chronological order).

    Returns:
        The number of rated matches.
    """
    Player.objects.update(
        games_played=0,
        wins=0,
        losses=0,
        draws=0,
        elo_rating=DEFAULT_RATING,
        bayeselo_rating=DEFAULT_RATING,
    )
    RatingHistory.objects.all().delete()

    applied = 0
    for game in Game.objects.filter(status=GameStatus.FINISHED).order_by("finished_at", "id"):
        if apply_ratings_for_game(game):
            applied += 1
    return applied
