#!/bin/bash
set -euo pipefail
cd /home/liar/ml/PAGraphomer

exec > >(tee /tmp/mcl42.log) 2>&1

echo "===== BBBP seed=42 molmcl ====="
pixi run python main.py --cfg configs/OTFormer/finetune/ogbg-molbbbp-OTFormer-finetune-molmcl.yaml wandb.use False

echo "===== Clintox seed=42 molmcl ====="
pixi run python main.py --cfg configs/OTFormer/finetune/ogbg-molclintox-OTFormer-finetune-molmcl.yaml wandb.use False

echo "===== BACE seed=42 molmcl ====="
pixi run python main.py --cfg configs/OTFormer/finetune/ogbg-molbace-OTFormer-finetune-molmcl.yaml wandb.use False

echo "===== ALL DONE ====="
