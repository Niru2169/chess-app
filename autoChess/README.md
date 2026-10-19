# AutoChess (Django + HTMX)

A small web application where **two pre-trained chess engines** play against each other.
The UI is intentionally minimal: left engine panel, center chessboard, right engine panel, plus a move list and match history.

This repository is generated to follow the **AutoChess** specification you provided. 

## Key characteristics

- **Backend**: Django
- **Frontend**: HTMX (no SPA build pipeline)
- **Engines**: UCI engines (MVP: Stockfish vs Stockfish)
- **Execution model**: A `tick` endpoint advances the game by exactly **one ply** (half-move) under a DB row lock.
- **Speed**: UI speed is a **move interval** (`move_interval_ms`), not engine think-time. 
- **Draw policy**: if a draw is claimable, the system **auto-claims** it to avoid infinite games. 

## Using python-chess (SVG board and pieces)

In the UI the board and pieces are rendered **as SVG** using the `chess.svg` module of the `python-chess` (`chess`) package. This gives stable, platform-independent rendering (it does not depend on the system fonts) and supports visual highlighting of the last move and check.

Important: `python-chess` is licensed under **GPLv3**. If licensing is strict for your project, only use this dependency if the copyleft terms are acceptable to you. (See `third_party_licences.md`.)

## Quick start (Windows 11 / Linux)

This project loads `.env` automatically via **python-dotenv** (see `manage.py`, `asgi.py`, `wsgi.py`).

1. Create venv and install deps:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS

   pip install -r requirements.txt
   ```

2. Install Stockfish (binary must be available on PATH or configured by env var)

   - Linux: install via your package manager (e.g. `apt install stockfish`), or download from Stockfish official site.
   - Windows: download a Stockfish build and set `AUTOCHESS_STOCKFISH_PATH` to the exe path.

3. Configure environment:

   ```bash
   copy .env.example .env  # Windows
   # cp .env.example .env   # Linux
   ```

   Then edit `.env` (especially `AUTOCHESS_STOCKFISH_PATH` if needed).

4. Migrate and run:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

Open: http://127.0.0.1:8000/

## Summary

This is a Django + HTMX web app where two UCI chess engines (by default Stockfish vs Stockfish) play against each other.
The game is advanced through a **tick** endpoint: one tick = one half-move, with a transaction and a row lock.
The "speed" is the **move interval**, not the think time.

## PGN viewer / review

Every match (engine or hot-seat) can be stepped through at `GET /games/<id>/review/`: the board is
rebuilt from the stored UCI moves at any ply, with **start / previous / next / end** buttons, a
clickable move list and ← → / Home / End keyboard shortcuts. The board fragment is swapped in by
HTMX. Reach it from the **Review** button on the match page.

## GNN prediction

The match page shows a **live GNN prediction** panel for the current position, powered by the graph
neural network in `../BayesianELO-main`. Instead of a fixed number of "top N" moves it lists the
**nucleus**: the smallest set of highest-probability moves whose cumulative probability reaches
`top-p` (default 0.9 = the top 90 % of the policy). It also shows the model's predicted result
(white / draw / black).

The same model is also selectable as an **engine side** (engine type `GNN (graph neural net)` in each
side panel), so the GNN can play matches, be rated (Elo + BayesElo) and export PGN like any other
engine. Without Stockfish installed, moves are applied with the python-chess fallback so
engine-vs-engine games still run.

**Search depth.** A GNN side can play in two ways, chosen with the panel's **Search depth** control:

| Setting | Behaviour |
| --- | --- |
| `Policy only (top-p)` | No search — a move is sampled from the nucleus (top-p) policy. |
| `Depth 3 / 5 / 8` | **Minimax search** over the GNN's policy + value heads (the tree is built breadth-first and evaluated in batches on the GPU). |

Deeper searches see further ahead; they stay fast because the tree is pruned by cumulative path
probability (`AUTOCHESS_GNN_MIN_PROB`, default `0.001`). Depth 8 typically takes a few tenths of a
second per move on a laptop GPU.

Each depth is a **separate rated player** (`GNN (policy)`, `GNN (depth 3)`, …), so the leaderboard
can compare depths directly.

- Config (`.env`): `AUTOCHESS_GNN_ENABLED`, `AUTOCHESS_GNN_DIR`, `AUTOCHESS_GNN_MODEL`,
  `AUTOCHESS_GNN_TOP_P`, `AUTOCHESS_GNN_REFRESH_MS`, `AUTOCHESS_GNN_MIN_PROB`.
- Degrades gracefully: if `torch` is missing, or the checkpoint is absent, or the checkpoint no
  longer matches the model (e.g. after an architecture change), the panel shows a short one-line
  message instead of failing; the app runs fine without the GNN.
- Endpoint: `GET /games/<id>/prediction/` (HTMX fragment).

## License

GPL-3.0-only (project code). External components are listed in `docs/third_party_licenses.md`.



## Demo deployment notes (shared hosting / Hetzner Cloud)

This project **can** run on a small VM without Docker. For a demo with light traffic, keep it simple:

- Prefer **PostgreSQL** for production-like usage.
- If you keep **SQLite** (ok for demos), this repo enables:
  - `PRAGMA journal_mode=WAL` + `busy_timeout` (best effort) on startup,
  - `timeout=20s` on the Django SQLite connection,
  - a **per-game tick lock** (`Game.tick_lock`) to avoid double-tick races from multiple tabs.

### Recommended process model (no Docker)

Run a single app process to minimize SQLite contention:

- Gunicorn: `--workers 1 --threads 4` (or even `--threads 2`)
- Put it behind Nginx (optional for a demo).
- Use `collectstatic` and serve static files via Nginx (or Whitenoise if you prefer).

If you later observe real concurrency, migrate to PostgreSQL and scale workers normally.
