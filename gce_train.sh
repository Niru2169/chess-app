#!/usr/bin/env bash
# =============================================================================
# gce_train.sh — set up a GCE GPU VM and train 3 GNN models (one per distill mode)
#
# Run this ON the VM (e.g. gcloud compute ssh gnn-train -- -t, then bash gce_train.sh).
# It installs the NVIDIA driver + Python deps, clones the repo, extracts the
# CCRL-4040 dataset, preprocesses it, and launches 3 trainings in parallel
# (one per GPU): --distill-mode minimax / mcts / both.
#
# Expect ~8-12h for all three at ~$29/hr ≈ $250-350 total on an a2-highgpu-8g.
# =============================================================================
set -euo pipefail

# ----------------------------- config ---------------------------------------
ZONE="${ZONE:-us-central1-a}"
REPO_DIR="$HOME/chess"
ENGINE_DIR="$REPO_DIR/BayesianELO-main"
DATA_ARCHIVE="${DATA_ARCHIVE:-$HOME/CCRL-4040.[1967497].pgn.7z}"
DATA_DIR="$HOME/training_data"
PGN="${PGN:-$DATA_DIR/CCRL-4040.[1967497].pgn}"
SHARDS_DIR="$ENGINE_DIR/data"
VENV="$HOME/gnn-venv"

# training hyper-parameters (chosen for ~8-12h on one A100 for this tiny model)
SAMPLE_RATE="${SAMPLE_RATE:-0.3}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DISTILL_EVERY="${DISTILL_EVERY:-300}"
DISTILL_BOARDS="${DISTILL_BOARDS:-8}"
SIMULATIONS="${SIMULATIONS:-32}"
SUBGRAPHS="${SUBGRAPHS:-2}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-$(( $(nproc) - 4 ))}"

# ----------------------------- 1. NVIDIA driver ------------------------------
echo "=== [1/7] NVIDIA driver ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y linux-headers-$(uname -r) wget
  # CUDA keyring -> latest driver (supports the CUDA 12.x runtime torch ships)
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-drivers
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ----------------------------- 2. system deps --------------------------------
echo "=== [2/7] system deps + Python ==="
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip git p7zip-full tmux

# ----------------------------- 3. clone repo ---------------------------------
echo "=== [3/7] clone repo ==="
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules https://github.com/Niru2169/chess-app.git "$REPO_DIR"
else
  echo "  repo already present, updating..."
  git -C "$REPO_DIR" pull --recurse-submodules
fi
echo "  engine branch: $(git -C "$ENGINE_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$ENGINE_DIR" rev-parse --short HEAD)"

# ----------------------------- 4. dataset ------------------------------------
echo "=== [4/7] dataset ==="
mkdir -p "$DATA_DIR"
if [ ! -f "$PGN" ]; then
  # try to find an uploaded archive (match loosely; the filename contains [1967497])
  found="$(find "$HOME" /tmp -maxdepth 3 -name '*CCRL*4040*.7z' 2>/dev/null | head -n1 || true)"
  if [ -n "$found" ]; then
    echo "  extracting $found"
    7z x "$found" -o"$DATA_DIR" -y >/dev/null
  fi
fi
if [ ! -f "$PGN" ]; then
  cat <<'EOF'
  !!! dataset not found !!!

  Upload the CCRL-4040 archive to the VM, then re-run this script:

    gcloud compute scp "D:\ArchivedLargeStuff\Documents\chess\CCRL-4040.[1967497].pgn.7z" \
        gnn-train:"$HOME"/ --zone=us-central1-a

  (the script extracts any ~/CCRL-4040*.pgn.7z automatically)
EOF
  exit 1
fi
echo "  dataset: $PGN ($(du -h "$PGN" | cut -f1))"

# ----------------------------- 5. python env ---------------------------------
echo "=== [5/7] python env ==="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$ENGINE_DIR/requirements.txt"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ----------------------------- 6. preprocess ---------------------------------
echo "=== [6/7] preprocess (sample-rate $SAMPLE_RATE, $PREPROCESS_WORKERS workers) ==="
mkdir -p "$SHARDS_DIR"
python "$ENGINE_DIR/gnn-chess/preprocess.py" "$PGN" --out "$SHARDS_DIR" \
  --sample-rate "$SAMPLE_RATE" --workers "$PREPROCESS_WORKERS"
echo "  shards: $(find "$SHARDS_DIR" -name 'shard_*.npz' | wc -l)"

# ----------------------------- 7. train 3 models ------------------------------
echo "=== [7/7] train 3 models ==="
cd "$ENGINE_DIR"
mkdir -p "$ENGINE_DIR/runs"

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
echo "  GPUs available: $NGPU"

launch() {
  local gpu="$1" mode="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python gnn-chess/train.py \
    --ignore_duplicates $IGNORE_DUPLICATES \
    --data "$SHARDS_DIR" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --device cuda --save "$ENGINE_DIR/runs/model_${mode}.pt" \
    --distill-mode "$mode" \
    --distill-every "$DISTILL_EVERY" --distill-boards "$DISTILL_BOARDS" \
    --simulations "$SIMULATIONS" --subgraphs "$SUBGRAPHS" \
    > "$ENGINE_DIR/runs/train_${mode}.log" 2>&1
  local rc=$?
  echo "  [gpu $gpu] $mode finished (exit $rc) -> $ENGINE_DIR/runs/model_${mode}.pt"
  return "$rc"
}

rc=0
if [ "$NGPU" -ge 3 ]; then
  echo "  running minimax / mcts / both in parallel on GPUs 0,1,2"
  launch 0 minimax & p0=$!
  launch 1 mcts    & p1=$!
  launch 2 both    & p2=$!
  wait "$p0" || rc=1
  wait "$p1" || rc=1
  wait "$p2" || rc=1
else
  echo "  fewer than 3 GPUs - running the 3 modes sequentially on GPU 0"
  launch 0 minimax || rc=1
  launch 0 mcts    || rc=1
  launch 0 both    || rc=1
fi

echo "=== done. checkpoints: ==="
ls -lh "$ENGINE_DIR"/runs/*.pt
echo "logs: $ENGINE_DIR/runs/train_{minimax,mcts,both}.log"
exit "$rc"
