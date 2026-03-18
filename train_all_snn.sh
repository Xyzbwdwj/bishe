#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-_smoke}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-50000}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"

# Reduce host-memory pressure from BLAS/OpenMP thread pools.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$OUT_DIR"

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "[error] '$PY' cannot import torch. Activate your training env first."
  echo "        Example: conda activate bishe"
  exit 1
fi

COMMON_ARGS=(
  --epochs "$TOTAL_EPOCHS"
  --gpu "$GPU"
  --pred 1
  --snn 1
  --auto-snn-tune 0
  --adam 1
  --lr 0.001
  --lr_step 10000,20000,30000,40000
  --grad-clip 2.0
  --sg-beta 5
  --ac_output tanh
)

run_case() {
  local name="$1"
  local input="$2"
  shift 2

  local save_path="$OUT_DIR/$name"
  local ckpt="${save_path}.pth.tar"

  # Keep finished runs; rerun only missing/empty checkpoints.
  if [[ -s "$ckpt" ]]; then
    echo "[skip] $name already finished: $ckpt"
    return 0
  fi

  echo "[run] $name"
  "$PY" Main.py \
    "${COMMON_ARGS[@]}" \
    --input "$input" \
    --savename "$save_path" \
    "$@"
}

# 1) Small input: keep baseline settings.
run_case "snn_seq1_50k_re2" "data/SeqN1T100.pth.tar" \
  --hidden-n 200 \
  --print-freq 500

# 2) Ns200 input: keep baseline settings.
run_case "snn_ns200_50k_re2" "data/Ns200_SeqN100_1.pth.tar" \
  --hidden-n 200 \
  --print-freq 500

# 3) Marcus50 input (batch=50): anti-OOM settings.
# - Lower hidden size
# - Very sparse recording (cuts y_hat/hidden buffers)
# - Keep full-batch updates for speed (interleaved is too slow for 50k)
run_case "snn_marcus50_50k_re2" "data/InputNs50_SeqN100_StraightTraj_Marcus_v2.pth.tar" \
  --hidden-n 128 \
  --print-freq 10000 \
  --grad-clip 1.0

echo "Done."
