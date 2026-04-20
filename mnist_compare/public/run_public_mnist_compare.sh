#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-python3}"
EPOCHS="${EPOCHS:-3000}"
PRINT_FREQ="${PRINT_FREQ:-500}"
SEEDS="${SEEDS:-0 1 2}"
OUT_DIR="${OUT_DIR:-mnist_compare/public}"
MNIST_ROOT="${MNIST_ROOT:-data/mnist_torchvision}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MPLCONFIGDIR

mkdir -p "$OUT_DIR" "Elman_SGD/predloss/Rotated" "$MNIST_ROOT"

echo "[1/4] Ensure MNIST raw array files exist"
"$PY" - <<'PY'
import os
import numpy as np
from torchvision import datasets

root = os.environ.get("MNIST_ROOT", "data/mnist_torchvision")
os.makedirs(root, exist_ok=True)
train = datasets.MNIST(root=root, train=True, download=True)
os.makedirs("Elman_SGD/predloss/Rotated", exist_ok=True)
X = train.data.numpy().reshape(-1, 28 * 28).astype("float32") / 255.0
y = train.targets.numpy().astype("int64")
np.save("Elman_SGD/predloss/Rotated/MNIST_X_train.npy", X)
np.save("Elman_SGD/predloss/Rotated/MNIST_labels.npy", y)
print("Saved MNIST arrays:", X.shape, y.shape)
PY

echo "[2/4] Build PCA sequence input"
"$PY" Figure6_InputPrep.py --out Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5.pth.tar

echo "[3/4] Train RNN/SNN compare runs"
for seed in $SEEDS; do
  echo "  - seed=$seed RNN"
  "$PY" Main.py \
    --gpu 0 \
    --epochs "$EPOCHS" \
    -p "$PRINT_FREQ" \
    --input Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5.pth.tar \
    --pred 1 \
    --ac_output tanh \
    --adam 1 \
    --lr 0.001 \
    --seed "$seed" \
    --savename "$OUT_DIR/rnn_pred_seed${seed}_e${EPOCHS}"

  echo "  - seed=$seed SNN"
  "$PY" Main.py \
    --gpu 0 \
    --epochs "$EPOCHS" \
    -p "$PRINT_FREQ" \
    --input Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5.pth.tar \
    --pred 1 \
    --snn 1 \
    --auto-snn-tune 0 \
    --ac_output tanh \
    --adam 1 \
    --lr 0.001 \
    --sg-beta 5 \
    --grad-clip 2.0 \
    --seed "$seed" \
    --savename "$OUT_DIR/snn_pred_seed${seed}_e${EPOCHS}"
done

echo "[4/4] Evaluate and summarize"
"$PY" "$OUT_DIR/evaluate_public_mnist_compare.py" \
  --ckpt-dir "$OUT_DIR" \
  --input-meta Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5.pth.tar \
  --out-dir "$OUT_DIR/report"

echo "Done. See: $OUT_DIR/report/summary.md"
