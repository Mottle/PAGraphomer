#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(
  ogbg-molclintox
  ogbg-molmuv
  ogbg-molhiv
  ogbg-molbace
  ogbg-moltox21
  ogbg-molsider
)

SEEDS="${*:-0}"

for seed in $SEEDS; do
  for ds in "${DATASETS[@]}"; do
    cfg="configs/OTFormer/finetune/${ds}-OTFormer-finetune.yaml"
    echo "============================================================"
    echo "Dataset: $ds | Seed: $seed"
    echo "Start: $(date '+%F %T')"
    echo "============================================================"
    pixi run python main.py --cfg "$cfg" seed "$seed"
    echo "Finished: $ds (seed=$seed) at $(date '+%F %T')"
    echo
  done
done

echo "All finetuning runs completed."
