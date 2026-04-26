#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Usage:
#   bash figures/Figure4_run.sh
#   PY=/path/to/python bash figures/Figure4_run.sh
#   FULL_RUN=1 bash figures/Figure4_run.sh
#
# Defaults are a quick "smoke" run so Figure4.py can complete on CPU.
# Set FULL_RUN=1 for long/full training settings.

PY="${PY:-python3}"
if [[ -z "${GPU:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU=1
  else
    GPU=0
  fi
fi
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MPLCONFIGDIR
echo "[info] GPU mode: $GPU (set GPU=0/1 manually to override)"

if [[ "${FULL_RUN:-0}" == "1" ]]; then
  MAIN_S4_EPOCHS=50000
  PRED_EPOCHS=100000
  REMAP_N_EPOCHS=2000
  REMAP_F_EPOCHS=20000
  PRINT_FREQ=1000
else
  MAIN_S4_EPOCHS=2
  PRED_EPOCHS=2
  REMAP_N_EPOCHS=2
  REMAP_F_EPOCHS=2
  PRINT_FREQ=1
fi

OUT="Elman_SGD/Remap_predloss"
BASE="$OUT/N200T100_relu_fixio"
mkdir -p \
  "$BASE/stages" \
  "$BASE/F5per_stages" \
  "$BASE/F10per_stages" \
  "$BASE/F20per_stages" \
  "$BASE/F30per_stages" \
  "$BASE/F40per_stages" \
  "$BASE/F50per_stages"

echo "[1/4] Train Panel B/C models on CPU"
"$PY" train_predict/Main_s4.py \
  --gpu "$GPU" \
  --epochs "$MAIN_S4_EPOCHS" \
  -p "$PRINT_FREQ" \
  --input data/Ns200_SeqN100_1.pth.tar \
  --batch-size 1 \
  --net ElmanRNN_tp1 \
  --pred 1 \
  --fixi 1 \
  --savename "$OUT/Ns200_SeqN100_predloss_full"

"$PY" train_predict/Main_s4.py \
  --gpu "$GPU" \
  --epochs "$MAIN_S4_EPOCHS" \
  -p "$PRINT_FREQ" \
  --input data/Ns200_SeqN100_2Batch.pth.tar \
  --batch-size 2 \
  --net ElmanRNN_tp1 \
  --pred 1 \
  --fixi 1 \
  --savename "$OUT/Ns200_SeqN100_2Batch_predloss"

echo "[2/4] Train base remap model"
"$PY" train_predict/Main_clean.py \
  --gpu "$GPU" \
  --epochs "$PRED_EPOCHS" \
  -p "$PRINT_FREQ" \
  --input data/Ns200_SeqN100_1.pth.tar \
  --ae 1 \
  --fixi 2 \
  --fixo 2 \
  --pred 1 \
  --hidden-n 200 \
  --rnn_act relu \
  --ac_output sigmoid \
  --savename "$BASE/pred_relu"

echo "[3/4] Train F->N remap stage"
"$PY" train_predict/Main_clean.py \
  --gpu "$GPU" \
  --epochs "$REMAP_N_EPOCHS" \
  -p "$PRINT_FREQ" \
  --input data/Ns200_SeqN100_2.pth.tar \
  --resume "$BASE/pred_relu.pth.tar" \
  --ae 1 \
  --fixi 2 \
  --fixo 2 \
  --pred 1 \
  --hidden-n 200 \
  --rnn_act relu \
  --ac_output sigmoid \
  --savename "$BASE/stages/remap_s0"

echo "[4/4] Train F->F remap stages (5/10/20/30/40/50)"
for noise in 5 10 20 30 40 50; do
  "$PY" train_predict/Main_clean.py \
    --gpu "$GPU" \
    --epochs "$REMAP_F_EPOCHS" \
    -p "$PRINT_FREQ" \
    --input "data/Ns200_SeqN100_1_${noise}per.pth.tar" \
    --resume "$BASE/pred_relu.pth.tar" \
    --ae 1 \
    --fixi 2 \
    --fixo 2 \
    --pred 1 \
    --hidden-n 200 \
    --rnn_act relu \
    --clamp_norm 0.5 \
    --ac_output sigmoid \
    --savename "$BASE/F${noise}per_stages/remap_s0"
done

echo "Done. Next step:"
echo "  $PY figures/Figure4.py"
