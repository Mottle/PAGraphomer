#!/usr/bin/env bash

set -euo pipefail

# Run OTFormer ablation finetuning configs sequentially from project root.
#
# Usage:
#   bash run/run_otformer_ablation_finetune.sh
#   bash run/run_otformer_ablation_finetune.sh bace
#   bash run/run_otformer_ablation_finetune.sh tox21
#   bash run/run_otformer_ablation_finetune.sh all 0 1 2
#
# Positional args:
#   1) dataset group: all | bace | tox21   (default: all)
#   2+) seeds to run sequentially          (default: 0)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_GROUP="${1:-all}"
shift || true

if [[ $# -eq 0 ]]; then
    SEEDS=(0)
else
    SEEDS=("$@")
fi

COMMON_CMD=(pixi run python main.py)

CONFIGS_BACE=(
    "configs/OTFormer/ablation_finetune/ogbg-molbace-OTFormer-full-readout.yaml"
    "configs/OTFormer/ablation_finetune/ogbg-molbace-OTFormer-no-rum-ot-node-pair-readout.yaml"
    "configs/OTFormer/ablation_finetune/ogbg-molbace-OTFormer-no-rum-ot-node-only-readout.yaml"
)

CONFIGS_TOX21=(
    "configs/OTFormer/ablation_finetune/ogbg-moltox21-OTFormer-full-readout.yaml"
    "configs/OTFormer/ablation_finetune/ogbg-moltox21-OTFormer-no-rum-ot-node-pair-readout.yaml"
    "configs/OTFormer/ablation_finetune/ogbg-moltox21-OTFormer-no-rum-ot-node-only-readout.yaml"
)

case "$DATASET_GROUP" in
    all)
        CONFIGS=("${CONFIGS_BACE[@]}" "${CONFIGS_TOX21[@]}")
        ;;
    bace)
        CONFIGS=("${CONFIGS_BACE[@]}")
        ;;
    tox21)
        CONFIGS=("${CONFIGS_TOX21[@]}")
        ;;
    *)
        echo "Unsupported dataset group: $DATASET_GROUP"
        echo "Choose one of: all, bace, tox21"
        exit 1
        ;;
esac

echo "Project root: $ROOT_DIR"
echo "Dataset group: $DATASET_GROUP"
echo "Seeds: ${SEEDS[*]}"
echo "Total configs: ${#CONFIGS[@]}"
echo

run_one() {
    local cfg_file="$1"
    local seed="$2"

    if [[ ! -f "$cfg_file" ]]; then
        echo "Missing config: $cfg_file"
        exit 1
    fi

    echo "============================================================"
    echo "Running config: $cfg_file"
    echo "Seed: $seed"
    echo "Start: $(date '+%F %T')"
    echo "Command: ${COMMON_CMD[*]} --cfg $cfg_file seed $seed"
    echo "============================================================"

    "${COMMON_CMD[@]}" --cfg "$cfg_file" seed "$seed"

    echo "Finished: $cfg_file (seed=$seed) at $(date '+%F %T')"
    echo
}

for seed in "${SEEDS[@]}"; do
    for cfg_file in "${CONFIGS[@]}"; do
        run_one "$cfg_file" "$seed"
    done
done

echo "All requested OTFormer ablation finetuning runs completed."
