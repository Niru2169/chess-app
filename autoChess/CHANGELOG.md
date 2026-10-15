# Changelog

All notable changes to this project will be documented here.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

- Initial MVP: Stockfish vs Stockfish, Django + HTMX.
- Rating system (Elo + BayesElo):
  - `Player` and `RatingHistory` models; `EngineConfig.player` link.
  - Incremental Elo with adaptive K-factor.
  - Dependency-free BayesElo batch fit (Coulom model, gradient ascent).
  - Ratings update automatically when a game finishes (idempotent per game).
  - `rebuild_ratings` management command for full recomputation.
  - Leaderboard at `/players/` and player profile with an SVG trend chart at `/players/<id>/`.
  - Participant ratings shown on the game panels.
- Local multiplayer (hot-seat): two humans alternate on the same device; no accounts, online
  play or stored per-user profiles. Select "Human vs Human (hot-seat)" when creating a match.
- PGN export:
  - SAN movetext rebuilt with python-chess when a game finishes (`MatchRecord.pgn`).
  - `GET /games/<id>/pgn/` downloads a single game as PGN.
  - `GET /exports/pvp.pgn` bundles every finished human-vs-human match.
