# Architecture

## Overview

The system is intentionally "server-driven UI" (HTMX): the server renders HTML fragments and the browser swaps them into place.

Core concepts:
- **Game**: persisted state of a match (FEN, status, result, next action time).
- **EngineConfig**: per-side engine settings (strength + think-time).
- **Move**: per-ply records, including FEN after the ply.
- **MatchRecord**: denormalized summary for dashboards/history.

## State machine

- CONFIGURED → RUNNING → PAUSED → RUNNING → FINISHED

The `tick` endpoint:
- runs inside a DB transaction,
- acquires a row lock (`select_for_update()`),
- may or may not advance one ply depending on `next_action_at`.

## Engine integration

An engine is accessed via UCI. MVP assumes Stockfish.

There are two logical roles:
- **Mover engine** (per side): provides `bestmove` for current position.
- **Arbiter** (Stockfish): updates FEN via `position ... moves ...` + `d`, and provides:
  - `Fen:` line (new position),
  - `Checkers:` line (in-check detection),
  - `Key:` line (position key for repetition detection).

Note: Stockfish's `d` command is not part of the UCI standard, but is widely available in Stockfish builds.

## Concurrency model

This MVP assumes a single Django worker process (typical dev / simple deployment).
If you scale to multiple workers, per-process engine pools must be reconsidered (e.g., Celery/Redis worker model).

## Rating system

Rated participants are `Player` rows (a human may link a Django `User` via `Player.user`;
engines get an auto-created `Player` with `is_engine=True`). A side is linked to a rated
participant through `EngineConfig.player`.

When a game finishes, `arena.services.game_loop.finalize_game()` runs (also from the
human-move path in `views.py`):

1. `MatchRecord` is materialized (as before).
2. `arena.services.rating.apply_ratings_for_game()` updates ratings (idempotent per game).

Two independent systems are maintained:

- **Elo** — incremental, adaptive K-factor (40 for new players, 20 default, 10 above 2400).
- **BayesElo** — a batch fit over every rated finished game (Coulom model with a draw
  parameter `kappa`). It is refit after each finished game and stored on `Player.bayeselo_rating`.

`arena.services.rating` has no third-party dependencies (no scipy/numpy): the BayesElo
log-posterior (concave) is maximized with gradient ascent + backtracking.

`RatingHistory` stores a per-game snapshot (Elo + BayesElo and their deltas). The
`rebuild_ratings` management command recomputes everything from scratch in chronological
order — useful after changing the rating parameters:

```
python manage.py rebuild_ratings
```

Settings: `AUTOCHESS_RATE_ENGINES` (auto-register engine participants) and
`AUTOCHESS_BAYESELO_PRIOR_STD` (BayesElo prior strength).

UI: `/players/` (leaderboard) and `/players/<id>/` (profile + inline SVG rating chart).
Participant ratings are shown on the game panels.

## Multiplayer (local hot-seat)

Multiplayer is intentionally offline and account-free: two people take turns on the same
device. A match is created in "Human vs Human (hot-seat)" mode (both `EngineConfig` rows have
`engine_type=PLAYER`), and the move endpoint validates and applies clicks for whichever side is
to move. No users, logins or per-user histories are involved — the point is to produce
human-vs-human games that can be exported and, together with engine games, feed the rating model.

## PGN export

When a game finishes, `finalize_game()` writes a PGN into `MatchRecord.pgn`
(`arena.services.pgn.build_pgn`). The SAN movetext is rebuilt from the stored UCI moves with
python-chess, so the export is valid even though the live loop records UCI.

- `GET /games/<id>/pgn/` — one game as a downloadable PGN file.
- `GET /exports/pvp.pgn` — all finished human-vs-human matches concatenated into one file
  (the primary multiplayer deliverable).
