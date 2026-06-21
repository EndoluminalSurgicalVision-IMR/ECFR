#!/bin/bash

# ============================================================================
# Learning-rate sweep used in the paper.
# The selected run is the one with the lowest Brier score in this fixed set.
# ============================================================================

: "${DATASET_CSV:?Set DATASET_CSV=/path/to/test.csv}"
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
SEED="${SEED:-2026}"
RESULTS_CSV="${RESULTS_CSV:-outputs/lr_sweep_${METHOD}_q${QUANTILE}.csv}"

LR_LIST=(1e-5 3.3e-5 6.6e-5 1e-4 3.3e-4 6.6e-4 1e-3)

echo "Starting learning-rate sweep for method: ${METHOD}"
echo "Search space: ${LR_LIST[*]}"
echo "Quantile: ${QUANTILE}"
echo "Adapted parameters: ${ADAPT_PARAMS}"
echo "Results CSV: ${RESULTS_CSV}"
echo "---------------------------------------------------------"
mkdir -p "$(dirname "${RESULTS_CSV}")"
rm -f "${RESULTS_CSV}"

for lr in "${LR_LIST[@]}"; do
    echo "[LR sweep] Running ${METHOD} with LR = $lr"
    
    # Execute the adaptation script
    python main_adaptation.py \
        --dataset_csv "${DATASET_CSV}" \
        --data_root "${DATA_ROOT}" \
        --method "${METHOD}" \
        --lr "${lr}" \
        --input_size "${INPUT_SIZE}" \
        --demo_backbone "${DEMO_BACKBONE}" \
        --adapt_params "${ADAPT_PARAMS}" \
        --capacity "${CAPACITY}" \
        --warmup_size "${WARMUP_SIZE}" \
        --ema_alpha "${EMA_ALPHA}" \
        --restore_prob "${RESTORE_PROB}" \
        --quantile "${QUANTILE}" \
        --output_csv "${RESULTS_CSV}" \
        --no_plot \
        --seed "${SEED}"
        
    echo "Finished LR = $lr"
    echo "---------------------------------------------------------"
done

if [ ! -s "${RESULTS_CSV}" ]; then
    echo "No results were written to ${RESULTS_CSV}."
    exit 1
fi

python -c "import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8'))); best=min(rows, key=lambda r: float(r['brier'])); print(f\"Selected LR by lowest Brier: {best['lr']} | Brier={float(best['brier']):.6f} | Acc={float(best['accuracy']):.6f} | ECE={float(best['ece']):.6f}\")" "${RESULTS_CSV}"
echo "Learning-rate sweep completed."
