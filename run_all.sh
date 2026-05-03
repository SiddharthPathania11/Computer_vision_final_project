#!/usr/bin/env bash
# Runs the full experiment pipeline end-to-end.
# Override defaults with environment variables:
#   DATA_ROOT, RESULTS_DIR, EPOCHS, BATCH_SIZE, BACKBONE

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKBONE="${BACKBONE:-efficientnet_b0}"
PYTHON="${PYTHON:-python}"

echo "Starting pipeline: DATA_ROOT=$DATA_ROOT EPOCHS=$EPOCHS BACKBONE=$BACKBONE"

echo "[0/6] Preparing splits ..."
$PYTHON src/prepare_data.py --data-root "$DATA_ROOT"

for cfg in 1 2 3 4; do
    echo "[${cfg}/6] Training config ${cfg} ..."
    $PYTHON src/train.py \
        --config "$cfg" \
        --data-root "$DATA_ROOT" \
        --results-dir "$RESULTS_DIR" \
        --backbone "$BACKBONE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE"
done

echo "[5/6] Evaluating all configs ..."
$PYTHON src/evaluate.py \
    --configs 1 2 3 4 \
    --data-root "$DATA_ROOT" \
    --results-dir "$RESULTS_DIR" \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE"

echo "[6/6] Running robustness suite ..."
$PYTHON src/robustness.py \
    --configs 1 2 3 4 \
    --data-root "$DATA_ROOT" \
    --results-dir "$RESULTS_DIR" \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE"

echo "Done. Results written to $RESULTS_DIR/"
