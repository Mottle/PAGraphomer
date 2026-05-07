#!/bin/bash
set -euo pipefail
cd /home/liar/ml/PAGraphomer
exec > >(tee /tmp/otformer_mcl_s42_r3.log) 2>&1

for exp in \
  ogbg-molbbbp-OTFormer-finetune-molmcl \
  ogbg-molclintox-OTFormer-finetune-molmcl \
  ogbg-molbace-OTFormer-finetune-molmcl \
  ogbg-moltox21-OTFormer-finetune-molmcl \
  ogbg-molsider-OTFormer-finetune-molmcl \
  ogbg-molhiv-OTFormer-finetune-molmcl \
  ogbg-molmuv-OTFormer-finetune-molmcl; do
  cfg="configs/OTFormer/finetune/${exp}.yaml"
  echo "===== START ${exp} ====="
  rm -rf "results/finetune/${exp}/0" "results/finetune/${exp}/1" "results/finetune/${exp}/2"
  pixi run python main.py --cfg "$cfg" --repeat 3 wandb.use False
  echo "===== END ${exp} ====="
done

echo "===== ALL DONE ====="
