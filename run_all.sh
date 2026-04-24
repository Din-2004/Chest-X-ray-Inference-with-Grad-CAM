#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-sample_data}"
SAMPLE_IMAGES_DIR="${SAMPLE_IMAGES_DIR:-sample_images}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/best_resnet50.pt}"
EVAL_DIR="${EVAL_DIR:-eval_results}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
INPUT_SIZE="${INPUT_SIZE:-224}"
VAL_FRACTION="${VAL_FRACTION:-0.25}"

echo "[1/5] Installing requirements"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[2/5] Preparing synthetic sample subset (download stub)"
"$PYTHON_BIN" download_sample_subset.py \
  --output_dir "$DATA_DIR" \
  --sample_images_dir "$SAMPLE_IMAGES_DIR" \
  --seed "$SEED"

echo "[3/5] Training smoke-test model"
"$PYTHON_BIN" train.py \
  --data_dir "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --batch "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --seed "$SEED" \
  --input_size "$INPUT_SIZE" \
  --val_fraction "$VAL_FRACTION" \
  --workers 0 \
  --checkpoint "$CHECKPOINT_PATH" \
  --no_pretrained

echo "[4/5] Evaluating checkpoint"
"$PYTHON_BIN" eval.py \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output_dir "$EVAL_DIR" \
  --batch "$BATCH_SIZE" \
  --seed "$SEED" \
  --input_size "$INPUT_SIZE" \
  --val_fraction "$VAL_FRACTION" \
  --workers 0

echo "[5/5] Launching Streamlit on http://localhost:${STREAMLIT_PORT}"
"$PYTHON_BIN" -m streamlit run app.py \
  --server.port "$STREAMLIT_PORT" \
  --server.address 0.0.0.0 \
  --server.headless true
