# Chess — a unified chess application

This repository combines two pieces that together form the app:

| Folder | What it is |
| --- | --- |
| [`autoChess/`](autoChess/) | **Django + HTMX web app** — match runner, live board, local hot‑seat play, Elo + BayesElo ratings with a leaderboard, and PGN export. |
| [`BayesianELO-main/`](BayesianELO-main/) · upstream: [duckyies/BayesianELO](https://github.com/duckyies/BayesianELO) (BayesElo + `gnn-chess/`) · related: [duckyies/ChessGNN](https://github.com/duckyies/ChessGNN) | **GNN chess engine** (PyTorch) — a graph neural network that plays chess, with training, top‑p (nucleus) move selection and an optional shallow search. Kudos to Ducky for the upstream BayesElo estimator and GNN this folder is built on. |

---

## Features

- **Match runner** — engine‑vs‑engine games advanced one ply at a time through a `tick` endpoint.
- **Local hot‑seat** — two humans alternate on one device (no accounts, no login).
- **Ratings** — classical Elo *and* a dependency‑free BayesElo (Coulom model) fit, shown on a leaderboard at `/players/`.
- **PGN export** — every finished game exports as PGN; all hot‑seat games export as one bundle.
- **PGN viewer / review** — step through any game move‑by‑move at `/games/<id>/review/` (buttons + ← → / Home / End keys), alongside the PGN export.
- **GNN engine** — plays from a trained checkpoint; move selection uses **top‑p (nucleus)**: instead of a fixed number of top moves, it keeps the smallest highest‑probability set whose cumulative probability reaches `top-p` (e.g. the top 90% of the mass). Each GNN side can also play with **minimax search at depth 3, 5 or 8** (selectable in the side panel; each depth is its own rated player). Selectable as an engine side in matches, with a Stockfish‑free move‑application fallback.
- **Live GNN prediction** — the match page shows the model's nucleus moves for the current position and its predicted result (white / draw / black). Degrades gracefully when `torch`/the checkpoint is absent.

---

## Ambitions & intentions

**What BayesElo is.** Classical Elo is *sequential*: ratings are updated one game at a time, in the
order the games happened to be played, so the final numbers depend on that order and on who happened
to meet whom. **BayesElo** (Rémi Coulom's method — <https://www.remi-coulom.fr/Bayesian-Elo>) instead
fits *every* rating at once, as a single maximum‑likelihood / Bayesian estimate over the whole set of
results, with an explicit **draw parameter** (`kappa`). That distinction matters in engine chess,
where a large share of games are drawn: scoring a draw as half a win distorts the ratings, whereas
BayesElo models the probability of a draw directly. And because ratings only mean something relative
to a reference, players of known strength (Stockfish pinned to a `UCI_Elo`) serve as **anchors**.

**Where the neural engine comes from.** The `gnn-chess/` network ships inside Ducky's
[BayesianELO](https://github.com/duckyies/BayesianELO); his related GNN research lives in
[ChessGNN](https://github.com/duckyies/ChessGNN). It follows the approach described in:

> Zięba, Marcin. (2023). *Exploring the applicability of graph neural networks for chess engines.*

Concretely: the **64 squares are the graph's nodes**, connected by **piece mobility** (the movement
patterns of the piece standing on each square); each node carries a 12-feature vector (piece type,
colour, square colour, rank/file, en-passant), and the move policy is a softmax over the graph's
**edges** — the candidate moves.

**What this project sets out to do.**

- Make both rating ideas usable **live** rather than only as an offline batch job — Elo updates as
  each game finishes, and BayesElo is re‑fitted over the games played so far.
- Put everything in **one place**: watch engines play, play hot‑seat, step through the PGN, and see
  the ratings move.
- Treat the **GNN as a first‑class engine**, so it can be watched, played, exported and — most
  importantly — **rated** against Stockfish and against itself.
- Make **configurations comparable**: the same engine at a different search depth is a different
  rated player, so the leaderboard can answer "is depth 8 actually stronger than depth 3?".
- Keep the **web app dependency‑free**: the app reimplements the BayesElo fit
  (`autoChess/arena/services/rating.py`) without `scipy`/`numpy`, while the upstream CLI tools in
  `BayesianELO-main/` keep their scipy‑based fit for offline tournaments.

---

## Layout

```
autoChess/                     Django project
  arena/                       the app
    models.py                  Game, Move, EngineConfig, Player, RatingHistory
    services/rating.py         Elo + BayesElo
    services/game_loop.py      tick / finalize / rating application
    services/pgn.py            PGN + SAN builder, PvP bundle
    services/chart.py          inline SVG rating chart
    templates/arena/           HTML (HTMX fragments)
    management/commands/rebuild_ratings.py
  autochess_site/              settings, urls, wsgi/asgi
  docs/, ARCHITECTURE.md ...

BayesianELO-main/              GNN engine
  gnn-chess/                   plays.py, play.py, train.py, preprocess.py, model.py, features.py ...
    chessgnn.pt                trained checkpoint
  ours_gpu_extras/             chessutil, graph_builder, search, targets_and_loss, training_utils, data, labels
```

---

## A. Run the web app (`autoChess`)

Requires **Python 3.11+** (developed on 3.13). A Stockfish binary is only needed for engine‑vs‑engine and human‑vs‑engine games.

```powershell
cd autoChess
python -m venv .venv
.\.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)

pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env

python manage.py migrate
python manage.py runserver
```

Open <http://127.0.0.1:8000/>.

**Pages**

| URL | What |
| --- | --- |
| `/` | dashboard + match history |
| `/games/new` | start a match (Engine vs Engine, Human vs Engine, Human vs Human) |
| `/players/` | Elo + BayesElo leaderboard |
| `/players/<id>/` | player profile with rating history chart |
| `/games/<id>/review/` | step‑through viewer (board at any ply + move list) |
| `/games/<id>/pgn/` | download one game as PGN |
| `/exports/pvp.pgn` | download all hot‑seat games as one PGN |

**Notes**

- `.env` is loaded automatically. Keep `DJANGO_DEBUG=1` for local work.
- For engine play, set `AUTOCHESS_STOCKFISH_PATH` in `.env` to the Stockfish executable (or have `stockfish` on `PATH`).
- `AUTOCHESS_RATE_ENGINES=1` lets engines take part in the rating pool.
- Recompute every rating from scratch: `python manage.py rebuild_ratings`.
- Tests: `python manage.py test` (33 tests).

---

## B. Run the GNN engine (`BayesianELO-main`)

```powershell
# Install deps (from the repo root, into your venv)
pip install chess numpy
# GPU build of PyTorch (CPU works too, just slower):
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Play against the model** (you type moves in UCI, e.g. `e2e4`):

```powershell
cd BayesianELO-main\gnn-chess
python plays.py chessgnn.pt --model-plays black          # top-p (nucleus) sampling, default top-p 0.9
python plays.py chessgnn.pt --greedy                     # take the most probable candidate in the nucleus
python plays.py chessgnn.pt --search --depth 3           # minimax on top of the policy (3 / 5 / 8)
python plays.py chessgnn.pt --search --depth 5
python plays.py chessgnn.pt --search --depth 8
```

**Estimate the model's Elo vs Stockfish:**

```powershell
python play.py --stockfish C:\path\to\stockfish.exe --model chessgnn.pt --elo 1320 --games 10
```

**Train (optional)** — expects a PGN of games:

```powershell
python preprocess.py games.pgn --out data --sample-rate 0.1
python train.py --data data --epochs 3 --save chessgnn.pt
```

The engine's helper modules live in the sibling folder `..\ours_gpu_extras`.

---

## License

Project code is **GPL‑3.0-only**; bundled/GPL dependencies (e.g. `python-chess`) are listed in `autoChess/docs/third_party_licenses.md`.
