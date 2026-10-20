# Chess — a unified chess application

This repository combines two pieces that together form the app:

| Folder | What it is |
| --- | --- |
| [`autoChess/`](autoChess/) | **Django + HTMX web app** — match runner, live board, local hot‑seat play, Elo + BayesElo ratings with a leaderboard, and PGN export. |
| [`BayesianELO-main/`](BayesianELO-main/) · originally based on [duckyies/BayesianELO](https://github.com/duckyies/BayesianELO) (BayesElo + `gnn-chess/`) · related reading: [duckyies/ChessGNN](https://github.com/duckyies/ChessGNN) | **GNN chess engine** (PyTorch) — a graph neural network that plays chess, with training, top‑p (nucleus) move selection and an optional shallow search. Kudos to Ducky for the BayesElo estimator and GNN this folder was built from. |

---

## Features

- **Match runner** — engine‑vs‑engine games advanced one ply at a time through a `tick` endpoint.
- **Step back mid‑match** — on the match page, ⏮ ◀ ▶ walk the board through the moves played so far and **● Live** returns to following the game (it keeps running while you look back).
- **Local hot‑seat** — two humans alternate on one device (no accounts, no login).
- **Ratings** — classical Elo *and* a dependency‑free BayesElo (Coulom model) fit, shown on a leaderboard at `/players/` (participants appear once they have a rated game).
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
  (`autoChess/arena/services/rating.py`) without `scipy`/`numpy`, while the original CLI tools in
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

BayesianELO-main/              GNN engine — git submodule -> Niru2169/BayesianELO
  gnn-chess/                   plays.py, play.py, train.py, preprocess.py, model.py, features.py ...
    chessgnn.pt                trained checkpoint
  ours_gpu_extras/             chessutil, graph_builder, search, targets_and_loss, training_utils, data, labels
```

---

## Getting the code

The GNN engine lives in a git **submodule** — our own repo,
[`Niru2169/BayesianELO`](https://github.com/Niru2169/BayesianELO) — so fetch it too:

```powershell
git clone --recurse-submodules https://github.com/Niru2169/chess-app.git
cd chess-app
```

Already cloned without it? Initialise it in place:

```powershell
git submodule update --init
```

To pick up later changes in the engine (after they are pushed to
`Niru2169/BayesianELO`):

```powershell
git submodule update --remote BayesianELO-main   # latest commit from our own repo
```

Work on the engine inside `BayesianELO-main/` and commit/push it as its own repo.

### Training the engine

The engine's dependencies live in
[`BayesianELO-main/requirements.txt`](BayesianELO-main/requirements.txt) — install them into the
same venv as the app (or any Python 3.10+ environment) before training:

```powershell
pip install -r BayesianELO-main\requirements.txt
```

Retrain from a PGN database (the CCRL-4040 file is a 7z archive — extract it first, then point
`preprocess.py` at the resulting `.pgn`):

```powershell
cd BayesianELO-main
python gnn-chess\preprocess.py ..\training_data\games.pgn --out data --sample-rate 0.1 --workers 20
python gnn-chess\train.py --data data --epochs 3 --batch-size 256 --device cuda
```

`preprocess.py` writes resumable `.npz` shards (re-running skips shards that already exist,
deduplicates positions within each chunk, and `--min-elo`/`--max-games`/`--seed` are available).
`train.py` takes `--device cuda|cpu|xla` and the search/distillation flags below.

### Local / cloud training scripts

For convenience, `laptop_train.ps1` (Windows, RTX 4060) and `linux_train.sh` (native Linux)
run preprocess + the 3 distill modes (`minimax`, `mcts`, `both`) sequentially;
`gce_provision.sh` / `gce_train.sh` provision a GCE GPU VM. Override defaults with
`-SampleRate`/`-Epochs`/`-BatchSize` (PowerShell) or `SAMPLE_RATE=`/`EPOCHS=` env vars (sh).

While a run is going, `watch_train.ps1` (Windows) / `watch_train.sh` (Linux/macOS) tails its
log and draws a live progress bar, ETA and losses. It is read-only and never touches the
training process; Ctrl+C only closes the viewer. The run has to be writing to that file �?"
launch with `Tee-Object` / `tee` (see each script's header).

#### Engine architecture

The engine is a full Graph Network over the board graph (64 square nodes, piece-mobility edges):

- **Edge features** — every directed edge carries a 14-dim geometric/directional vector
  (rank/file offsets, Manhattan/Chebyshev distance, self-loop/knight-jump flags, 8-way compass).
- **CNN stem** — a 2-layer 3×3 residual conv block on the 8×8 grid captures local spatial
  patterns (pawn shields, adjacent squares) that have no mobility edge between them.
- **GATEAU blocks** — 10 residual blocks of edge-feature-aware multi-head graph attention
  (attention uses the `[e_ij, v_i, v_j]` triplet), with a global-to-node broadcast from the
  board state vector in every block.
- **Heads** — multi-task: `next_move` (policy over edges) and `final_result` (W/D/B) are the
  primary heads; `next_piece_moved/captured` and `plies_till_end` are useful auxiliaries;
  `legal_moves/captures` and the check heads are scaffold heads whose loss weight decays
  with training progress.
- **Search** — breadth-first minimax (depth 3/5/8) or lightweight virtual-loss batched PUCT
  MCTS, with an optional subgraph-sampling value ensemble to reduce leaf variance.
- **Distillation** — shallow-to-deep policy/value targets (Gumbel-minimax and/or MCTS visit
  distributions) refine the network during training.

```powershell
# train with search distillation and subgraph-ensemble evaluation
python gnn-chess\train.py --data data --epochs 3 --batch-size 256 --device cuda `
    --distill-mode both --distill-every 50 --distill-boards 8 --simulations 64 --subgraphs 2

# play against Stockfish with MCTS + a subgraph ensemble
python gnn-chess\play.py --stockfish stockfish.exe --model chessgnn.pt `
    --search-mode mcts --simulations 64 --subgraphs 2
```

The earlier GATEAU/MCTS experiments are parked on the `archive/gnn-training-2026-10-19` branch.

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
| `/games/<id>/` | live match (add `?ply=<n>` to show an earlier position) |
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
- Tests: `python manage.py test` (48 tests).

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
