#!/usr/bin/env bash
# =============================================================================
# watch_train.sh - live training dashboard for Linux/macOS (READ-ONLY).
#
# Tails a training log, parses the lines train.py / preprocess.py already print,
# and draws one screen with a progress bar, ETA and losses. It never writes to
# the log and never touches the training process: Ctrl+C only closes the viewer.
#
#   bash watch_train.sh                      # ./laptop_train.log
#   bash watch_train.sh -l runs/mcts.log
#   bash watch_train.sh -l runs/mcts.log -n  # skip the nvidia-smi line
#
# The training has to be writing to that file. If you started it in a plain
# terminal, redirect instead:
#   bash linux_train.sh 2>&1 | tee laptop_train.log
# then run this viewer in a second terminal.
# =============================================================================
set -uo pipefail

LOG="laptop_train.log"
NOGPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    -l|--log) LOG="$2"; shift 2 ;;
    -n|--no-gpu) NOGPU=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ helpers

fmt_num() { # 1234567 -> 1,234,567
  local s="${1:-}" out=""
  [ -z "$s" ] && { printf -- '--'; return; }
  while [ ${#s} -gt 3 ]; do
    out=",${s: -3}$out"
    s="${s:0:${#s}-3}"
  done
  printf '%s%s' "$s" "$out"
}

bar() { # percent(0-100) width
  local pct=$1 w=$2 fill i out=""
  fill=$(( pct * w / 100 ))
  for ((i = 0; i < w; i++)); do
    if [ "$i" -lt "$fill" ]; then out="$out#"; else out="$out."; fi
  done
  printf '[%s]' "$out"
}

fmt_dur() { # seconds -> 1:02:03 / 02:03
  local s=${1:-0}
  [ "$s" -lt 0 ] && s=0
  local h=$(( s / 3600 )) m=$(( (s % 3600) / 60 )) sec=$(( s % 60 ))
  if [ "$h" -gt 0 ]; then printf '%d:%02d:%02d' "$h" "$m" "$sec"
  else printf '%02d:%02d' "$m" "$sec"; fi
}

# -------------------------------------------------------------------- state

stage="waiting"
mode=""
step=0
total=0
loss=""
next_move=""
lr=""
val_loss=""
distill_loss=""
train_pos=""
val_pos=""
chunks=0
pre_pos=""
watch_start=$(date +%s)
last_step=0
last_step_time=0
rate_x1000=0
recent=""   # newline separated, newest last

add_event() {
  [ -z "${1:-}" ] && return
  recent="${recent}${1}"$'\n'
  recent=$(printf '%s' "$recent" | tail -n 5)
  recent="${recent}"$'\n'
}

render() {
  local now elapsed eta pct
  now=$(date +%s)
  elapsed=$(( now - watch_start ))
  printf '\033[H\033[2J'
  printf '\n  CHESS GNN TRAINING - live view (read-only)\n'
  printf '  %s\n' "--------------------------------------------------------------------------"
  printf '  log      : %s\n' "$LOG"
  printf '  stage    : %s%s\n' "$stage" "${mode:+   mode: $mode}"
  if [ "$NOGPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
    local g
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    [ -n "$g" ] && printf '  gpu      : %s\n' "$(printf '%s' "$g" | tr -d ' ' | tr ',' '/' )"
  fi
  printf '\n'

  if [ "$total" -gt 0 ]; then
    pct=$(( step * 100 / total ))
    if [ "$rate_x1000" -gt 0 ] && [ "$step" -lt "$total" ]; then
      eta=$(fmt_dur $(( (total - step) * 1000 / rate_x1000 )))
      rate_txt=$(awk -v r="$rate_x1000" 'BEGIN{printf "%.2f", r/1000}')
    else
      eta="--:--"; rate_txt="--"
    fi
    printf '  %s %6s%%   step %s / %s\n' "$(bar "$pct" 44)" "$(awk -v s="$step" -v t="$total" 'BEGIN{printf "%.1f", s*100/t}')" "$step" "$total"
    printf '  watching %s   eta %s   %s steps/s\n' "$(fmt_dur "$elapsed")" "$eta" "$rate_txt"
    printf '\n'
    printf '  loss %s   next_move %s   lr %s\n' "${loss:---}" "${next_move:---}" "${lr:---}"
    printf '  val loss %s   distill loss %s\n' "${val_loss:---}" "${distill_loss:---}"
    [ -n "$train_pos" ] && printf '  positions: %s train / %s val\n' "$(fmt_num "$train_pos")" "$(fmt_num "$val_pos")"
  elif [ "$stage" = "preprocess" ]; then
    printf '  preprocessing (no known total)\n'
    printf '  chunks %s   positions %s   watching %s\n' "$chunks" "$(fmt_num "$pre_pos")" "$(fmt_dur "$elapsed")"
  else
    printf '  waiting for progress lines...\n'
    if [ ! -f "$LOG" ]; then
      printf '\n  %s does not exist yet. Start the training with output redirected:\n' "$LOG"
      printf '    bash linux_train.sh 2>&1 | tee %s\n' "$LOG"
    fi
  fi

  printf '\n  recent:\n'
  if [ -z "$recent" ]; then printf '    (nothing yet)\n'; else
    printf '%s' "$recent" | sed '/^$/d' | sed 's/^/    /'
  fi
  printf '\n  Ctrl+C to close this viewer (training keeps running)\n'
}

parse() {
  local line="$1" ts delta dt
  if [[ "$line" =~ ^step\ ([0-9]+)/([0-9]+)\ +loss\ ([0-9.]+)\ +next_move\ ([0-9.]+)\ +lr\ ([^[:space:]]+) ]]; then
    ts=$(date +%s)
    local new_step=${BASH_REMATCH[1]}
    if [ "$last_step_time" -gt 0 ] && [ "$new_step" -gt "$last_step" ]; then
      dt=$(( ts - last_step_time ))
      if [ "$dt" -ge 1 ]; then
        delta=$(( new_step - last_step ))
        rate_x1000=$(( delta * 1000 / dt ))
      fi
    fi
    last_step=$new_step; last_step_time=$ts
    step=$new_step; total=${BASH_REMATCH[2]}
    loss=${BASH_REMATCH[3]}; next_move=${BASH_REMATCH[4]}; lr=${BASH_REMATCH[5]}
    return
  fi
  if [[ "$line" =~ ^[[:space:]]*\[val\]\ loss\ ([0-9.]+) ]]; then
    val_loss=${BASH_REMATCH[1]}; add_event "val loss ${BASH_REMATCH[1]}"; return
  fi
  if [[ "$line" =~ ^[[:space:]]*\[distill:([a-z]+)\]\ loss\ ([0-9.]+) ]]; then
    distill_loss=${BASH_REMATCH[2]}; return
  fi
  if [[ "$line" =~ ^---\ ([A-Za-z]+)\ ---[[:space:]]*$ ]]; then
    mode=${BASH_REMATCH[1]}
    step=0; total=0; loss=""; next_move=""; lr=""; val_loss=""; distill_loss=""
    last_step=0; last_step_time=0; rate_x1000=0
    stage="train"; add_event "started mode: ${BASH_REMATCH[1]}"; return
  fi
  if [[ "$line" =~ ^===\ (.+)\ ===[[:space:]]*$ ]]; then
    stage=${BASH_REMATCH[1]}; add_event "${BASH_REMATCH[1]}"; return
  fi
  if [[ "$line" =~ chunks=([0-9]+)\ positions=([0-9,]+) ]]; then
    chunks=${BASH_REMATCH[1]}; pre_pos=${BASH_REMATCH[2]//,/}; stage="preprocess"; return
  fi
  if [[ "$line" =~ train\ positions:\ ([0-9,]+)[[:space:]]+val\ positions:\ ([0-9,]+) ]]; then
    train_pos=${BASH_REMATCH[1]//,/}; val_pos=${BASH_REMATCH[2]//,/}
    add_event "loaded ${BASH_REMATCH[1]} train / ${BASH_REMATCH[2]} val positions"; return
  fi
  if [[ "$line" =~ ^saved\ (.+)[[:space:]]*$ ]]; then
    add_event "saved $(basename "${BASH_REMATCH[1]}")"; return
  fi
  if [[ "$line" =~ (Traceback|out\ of\ memory|FAILED) ]]; then
    add_event "$line"; return
  fi
}

# --------------------------------------------------------------------- main

trap 'printf "\033[?1049l"; exit 0' INT TERM
printf '\033[?1049h'   # alternate screen, restored on exit

if [ ! -e "$LOG" ]; then
  render
fi

while IFS= read -r line || [ -n "$line" ]; do
  parse "$line"
  render
done < <(tail -n 20 -F "$LOG" 2>/dev/null)
