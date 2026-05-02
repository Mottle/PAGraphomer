#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(
  ogbg-molbbbp
  ogbg-molclintox
  ogbg-molmuv
  ogbg-molhiv
  ogbg-molbace
  ogbg-moltox21
  ogbg-molsider
)

SEEDS="${*:-0}"

echo "Project root: $ROOT_DIR"
echo "Seeds: $SEEDS"
echo "Total datasets: ${#DATASETS[@]}"
echo

for seed in $SEEDS; do
  for ds in "${DATASETS[@]}"; do
    cfg="configs/OTFormer/finetune/${ds}-OTFormer-finetune.yaml"
    if [[ ! -f "$cfg" ]]; then
      echo "Missing config: $cfg"
      exit 1
    fi
    echo "============================================================"
    echo "Dataset: $ds | Seed: $seed"
    echo "Config: $cfg"
    echo "Start: $(date '+%F %T')"
    echo "============================================================"
    pixi run python main.py --cfg "$cfg" seed "$seed"
    echo "Finished: $ds (seed=$seed) at $(date '+%F %T')"
    echo
  done
done

echo "All finetuning runs completed."
