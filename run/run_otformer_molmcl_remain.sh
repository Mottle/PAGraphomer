#!/bin/bash
# Run remaining OTFormer molmcl finetune datasets sequentially
set -e
cd /home/liar/ml/PAGraphomer

for ds in muv hiv bace tox21 sider; do
    echo "===== Starting ogbg-mol${ds} ====="
    rm -rf "results/finetune/ogbg-mol${ds}-OTFormer-finetune-molmcl"
    pixi run python main.py \
        --cfg "configs/OTFormer/finetune/ogbg-mol${ds}-OTFormer-finetune-molmcl.yaml" \
        wandb.use False
    echo "===== Finished ogbg-mol${ds} ====="
done

echo "===== ALL DONE ====="
