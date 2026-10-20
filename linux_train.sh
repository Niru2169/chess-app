#!/usr/bin/env bash
# =============================================================================
# linux_train.sh — train the 3 distill modes on NATIVE Linux (dual-boot).
#
# Installs the NVIDIA driver if needed, builds a venv, preprocesses the CCRL
# dataset, and runs the 3 modes sequentially on the GPU.
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
IGNORE_DUPLICATES="${IGNORE_DUPLICATES:-}"

echo "=== [1/5] system deps + NVIDIA driver ==="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git p7zip-full tmux curl

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  installing NVIDIA driver..."
  sudo apt-get install -y linux-headers-"$(uname -r)"
  if grep -qi ubuntu /etc/os-release; then
    CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-noble}")"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  else
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  fi
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-drivers
  echo "  driver installed; if nvidia-smi still fails, reboot once."
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [2/5] repo ==="
if [ ! -d "$REPO/.git" ]; then
  git clone --recurse-submodules https://github.com/Niru2169/chess-app.git "$REPO"
else
  git -C "$REPO" pull --recurse-submodules
fi
echo "  engine: $(git -C "$ENGINE" rev-parse --abbrev-ref HEAD) @ $(git -C "$ENGINE" rev-parse --short HEAD)"

echo "=== [3/5] dataset ==="
mkdir -p "$DATA_DIR"
# search the Linux home plus any mounted Windows partition (/mnt, /media, /run/media)
SEARCH=("$DATA_DIR" "$HOME" /mnt/* /media/* /run/media/*/*)
PGN="$(find "${SEARCH[@]}" -maxdepth 4 -name 'CCRL-4040*.pgn' 2>/dev/null | head -n1 || true)"
if [ -z "$PGN" ]; then
  ARCHIVE="$(find "${SEARCH[@]}" -maxdepth 4 -name 'CCRL-4040*.7z' 2>/dev/null | head -n1 || true)"
  if [ -n "$ARCHIVE" ]; then
    echo "  extracting $ARCHIVE"
    7z x "$ARCHIVE" -o"$DATA_DIR" -y >/dev/null
    PGN="$(find "$DATA_DIR" -maxdepth 2 -name 'CCRL-4040*.pgn' | head -n1)"
  fi
fi
if [ -z "$PGN" ]; then
  echo "  dataset not found. Mount the Windows partition and copy it in, e.g.:"
  echo "    sudo mkdir -p /mnt/win && sudo mount /dev/nvme0n1p3 /mnt/win   # your Windows partition"
  echo "    cp '/mnt/win/ArchivedLargeStuff/Documents/chess/CCRL-4040.[1967497].pgn' '$DATA_DIR/'"
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
    --ignore_duplicates $IGNORE_DUPLICATES \
    --device cuda --save "$RUNS/model_$mode.pt" \
    --distill-mode "$mode" --distill-every "$DISTILL_EVERY" \
    --distill-boards "$DISTILL_BOARDS" --simulations "$SIMULATIONS" --subgraphs "$SUBGRAPHS"
done

echo "=== done. checkpoints: ==="
ls -lh "$RUNS"/*.pt
