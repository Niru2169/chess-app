#!/usr/bin/env bash
# =============================================================================
# linux_train.sh — train the 3 distill modes on Linux (WSL2 or a Linux host).
#
# In WSL2 the GPU is the Windows NVIDIA driver (nvidia-smi works) and pip's
# torch is the CUDA build. Same script works on a plain Linux box.
#
#   bash linux_train.sh
#   SAMPLE_RATE=0.05 EPOCHS=5 bash linux_train.sh
# =============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/chess}"
ENGINE="$REPO/BayesianELO-main"
VENV="${VENV:-$HOME/gnn-venv}"
DATA_DIR="${DATA_DIR:-$HOME/training_data}"
SHARDS="$ENGINE/data"
RUNS="$ENGINE/runs"

SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DISTILL_EVERY="${DISTILL_EVERY:-300}"
DISTILL_BOARDS="${DISTILL_BOARDS:-8}"
SIMULATIONS="${SIMULATIONS:-32}"
SUBGRAPHS="${SUBGRAPHS:-2}"

echo "=== [1/5] system deps ==="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git p7zip-full tmux

echo "=== [2/5] repo ==="
if [ ! -d "$REPO/.git" ]; then
  git clone --recurse-submodules https://github.com/Niru2169/chess-app.git "$REPO"
else
  git -C "$REPO" pull --recurse-submodules
fi
echo "  engine: $(git -C "$ENGINE" rev-parse --abbrev-ref HEAD) @ $(git -C "$ENGINE" rev-parse --short HEAD)"

echo "=== [3/5] dataset ==="
mkdir -p "$DATA_DIR"
# the Windows drive is /mnt/d/... in WSL; search the common spots (brackets in the
# filename mean we match loosely)
SEARCH=("$DATA_DIR" "$HOME" "/mnt/d/ArchivedLargeStuff/Documents/chess")
PGN="$(find "${SEARCH[@]}" -maxdepth 3 -name 'CCRL-4040*.pgn' 2>/dev/null | head -n1 || true)"
if [ -z "$PGN" ]; then
  ARCHIVE="$(find "${SEARCH[@]}" -maxdepth 3 -name 'CCRL-4040*.7z' 2>/dev/null | head -n1 || true)"
  if [ -n "$ARCHIVE" ]; then
    echo "  extracting $ARCHIVE"
    7z x "$ARCHIVE" -o"$DATA_DIR" -y >/dev/null
    PGN="$(find "$DATA_DIR" -maxdepth 2 -name 'CCRL-4040*.pgn' | head -n1)"
  fi
fi
if [ -z "$PGN" ]; then
  echo "  dataset not found. Copy it in, e.g.:"
  echo "    cp '/mnt/d/.../CCRL-4040.[1967497].pgn' '$DATA_DIR/'"
  exit 1
fi
echo "  dataset: $PGN ($(du -h "$PGN" | cut -f1))"

echo "=== [4/5] python env ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$ENGINE/requirements.txt"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [5/5] preprocess + train (sample-rate $SAMPLE_RATE, epochs $EPOCHS) ==="
mkdir -p "$SHARDS" "$RUNS"
python "$ENGINE/gnn-chess/preprocess.py" "$PGN" --out "$SHARDS" \
  --sample-rate "$SAMPLE_RATE" --workers "$(($(nproc) - 4))"
echo "  shards: $(find "$SHARDS" -name 'shard_*.npz' | wc -l)"

cd "$ENGINE"
for mode in minimax mcts both; do
  echo "--- $mode ---"
  python gnn-chess/train.py --data "$SHARDS" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --device cuda --save "$RUNS/model_$mode.pt" \
    --distill-mode "$mode" --distill-every "$DISTILL_EVERY" \
    --distill-boards "$DISTILL_BOARDS" --simulations "$SIMULATIONS" --subgraphs "$SUBGRAPHS"
done

echo "=== done. checkpoints: ==="
ls -lh "$RUNS"/*.pt
