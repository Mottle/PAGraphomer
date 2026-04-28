#!/usr/bin/env bash

set -euo pipefail

# Run ScaledRangeFormer supervised TU graph classification experiments on a
# single workstation GPU (e.g. RTX 4090 24G).
#
# Usage examples:
#   bash run/run_scaled_range_former_tu.sh
#   bash run/run_scaled_range_former_tu.sh NCI1
#   bash run/run_scaled_range_former_tu.sh PROTEINS Mixed 0 1 2
#   bash run/run_scaled_range_former_tu.sh DD all 0
#
# Positional args:
#   1) dataset group: all | NCI1 | PROTEINS | DD        (default: all)
#   2) variant: all | Shared | Mixed | Mixed-B+ | Sequential | Bypass (default: all)
#   3+) seeds to run sequentially                       (default: 0)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_GROUP="${1:-all}"
VARIANT_GROUP="${2:-all}"
shift 2 || true

if [[ $# -eq 0 ]]; then
    SEEDS=(0)
else
    SEEDS=("$@")
fi

COMMON_CMD=(pixi run python main.py)

cfg_path() {
    local dataset="$1"
    local variant="$2"
    echo "configs/ScaledRangeFormer/${dataset}-ScaledRangeFormer${variant}.yaml"
}

VARIANTS_ALL=(
    "Shared"
    "Mixed"
    "Mixed-B+"
    "Sequential"
    "Bypass"
)

resolve_variants() {
    local variant_group="$1"
    case "$variant_group" in
        all)
            printf '%s\n' "${VARIANTS_ALL[@]}"
            ;;
        Shared|Mixed|Mixed-B+|Sequential|Bypass)
            printf '%s\n' "$variant_group"
            ;;
        *)
            echo "Unsupported variant group: $variant_group" >&2
            echo "Choose one of: all, Shared, Mixed, Mixed-B+, Sequential, Bypass" >&2
            exit 1
            ;;
    esac
}

resolve_datasets() {
    local dataset_group="$1"
    case "$dataset_group" in
        all)
            printf '%s\n' "NCI1" "PROTEINS" "DD"
            ;;
        NCI1|PROTEINS|DD)
            printf '%s\n' "$dataset_group"
            ;;
        *)
            echo "Unsupported dataset group: $dataset_group" >&2
            echo "Choose one of: all, NCI1, PROTEINS, DD" >&2
            exit 1
            ;;
    esac
}

readarray -t DATASETS < <(resolve_datasets "$DATASET_GROUP")
readarray -t VARIANTS < <(resolve_variants "$VARIANT_GROUP")

echo "Project root: $ROOT_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

run_one() {
    local cfg_file="$1"
    local seed="$2"

    if [[ ! -f "$cfg_file" ]]; then
        echo "Missing config: $cfg_file" >&2
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
    for dataset in "${DATASETS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            run_one "$(cfg_path "$dataset" "$variant")" "$seed"
        done
    done
done

echo "All requested ScaledRangeFormer TU runs completed."
