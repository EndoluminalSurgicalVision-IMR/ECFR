#!/bin/bash

# Minimal single-run example after selecting a learning rate.
# Paper-style reporting should first run scripts/run_lr_sweep.sh and select
# the learning rate with the lowest Brier score from the predefined search set.

: "${DATASET_CSV:?Set DATASET_CSV=/path/to/test.csv}"
: "${LR:?Set LR to the selected value from scripts/run_lr_sweep.sh}"
METHOD="${METHOD:-ecfr}" # Options: ecfr, grata
QUANTILE="${QUANTILE:-0.5}" # ECFR-0.5: 0.5; ECFR-Acc: prior target accuracy, e.g., 0.83
DATA_ROOT="${DATA_ROOT:-}"
INPUT_SIZE="${INPUT_SIZE:-224}"
DEMO_BACKBONE="${DEMO_BACKBONE:-vit_b_16}"
ADAPT_PARAMS="${ADAPT_PARAMS:-layernorm}" # Paper setting: layernorm
CAPACITY="${CAPACITY:-128}"
EMA_ALPHA="${EMA_ALPHA:-0.999}"
RESTORE_PROB="${RESTORE_PROB:-0.01}"
WARMUP_SIZE="${WARMUP_SIZE:-5}"
PLOT_PATH="${PLOT_PATH:-outputs/ECFR_Quadrant_Flow.png}"
SEED="${SEED:-2026}"

python main_adaptation.py \
  --dataset_csv "${DATASET_CSV}" \
  --data_root "${DATA_ROOT}" \
  --method "${METHOD}" \
  --lr "${LR}" \
  --input_size "${INPUT_SIZE}" \
  --demo_backbone "${DEMO_BACKBONE}" \
  --adapt_params "${ADAPT_PARAMS}" \
  --capacity "${CAPACITY}" \
  --warmup_size "${WARMUP_SIZE}" \
  --ema_alpha "${EMA_ALPHA}" \
  --restore_prob "${RESTORE_PROB}" \
  --quantile "${QUANTILE}" \
  --plot_path "${PLOT_PATH}" \
  --seed "${SEED}"
