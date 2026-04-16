#!/usr/bin/env bash

set -euo pipefail

# Run GRIT downstream finetuning from the ZINC MotiL-style pretrained backbone.
#
# Usage:
#   bash run/run_grit_motil_finetune.sh
#   bash run/run_grit_motil_finetune.sh hiv
#   bash run/run_grit_motil_finetune.sh cls 0 1 2
#   bash run/run_grit_motil_finetune.sh reg 0
#
# Positional args:
#   1) dataset group: all | cls | reg | hiv | bace | bbbp | clintox | muv | tox21 | sider | esol | freesolv
#      default: all
#   2+) seeds to run sequentially, default: 0

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

CONFIGS_CLS=(
    "configs/GRIT/finetune/ogbg-molhiv-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-molbace-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-molbbbp-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-molclintox-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-molmuv-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-moltox21-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-molsider-GRIT-motil-finetune.yaml"
)

CONFIGS_REG=(
    "configs/GRIT/finetune/ogbg-molesol-GRIT-motil-finetune.yaml"
    "configs/GRIT/finetune/ogbg-molfreesolv-GRIT-motil-finetune.yaml"
)

case "$DATASET_GROUP" in
    all)
        CONFIGS=("${CONFIGS_CLS[@]}" "${CONFIGS_REG[@]}")
        ;;
    cls)
        CONFIGS=("${CONFIGS_CLS[@]}")
        ;;
    reg)
        CONFIGS=("${CONFIGS_REG[@]}")
        ;;
    hiv)
        CONFIGS=("configs/GRIT/finetune/ogbg-molhiv-GRIT-motil-finetune.yaml")
        ;;
    bace)
        CONFIGS=("configs/GRIT/finetune/ogbg-molbace-GRIT-motil-finetune.yaml")
        ;;
    bbbp)
        CONFIGS=("configs/GRIT/finetune/ogbg-molbbbp-GRIT-motil-finetune.yaml")
        ;;
    clintox)
        CONFIGS=("configs/GRIT/finetune/ogbg-molclintox-GRIT-motil-finetune.yaml")
        ;;
    muv)
        CONFIGS=("configs/GRIT/finetune/ogbg-molmuv-GRIT-motil-finetune.yaml")
        ;;
    tox21)
        CONFIGS=("configs/GRIT/finetune/ogbg-moltox21-GRIT-motil-finetune.yaml")
        ;;
    sider)
        CONFIGS=("configs/GRIT/finetune/ogbg-molsider-GRIT-motil-finetune.yaml")
        ;;
    esol)
        CONFIGS=("configs/GRIT/finetune/ogbg-molesol-GRIT-motil-finetune.yaml")
        ;;
    freesolv)
        CONFIGS=("configs/GRIT/finetune/ogbg-molfreesolv-GRIT-motil-finetune.yaml")
        ;;
    *)
        echo "Unsupported dataset group: $DATASET_GROUP"
        echo "Choose one of: all, cls, reg, hiv, bace, bbbp, clintox, muv, tox21, sider, esol, freesolv"
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

echo "All requested GRIT MotiL finetuning runs completed."
