#!/bin/bash
set -euo pipefail
cd /home/liar/ml/PAGraphomer
exec > >(tee /tmp/gcn_gine_gps_mcl_repeat3.log) 2>&1
for cfg in \
  configs/GCN/finetune/ogbg-molbbbp-GCN-mcl.yaml \
  configs/GCN/finetune/ogbg-moltox21-GCN-mcl.yaml \
  configs/GCN/finetune/ogbg-molbace-GCN-mcl.yaml \
  configs/GINE/finetune/ogbg-molbbbp-GINE-mcl.yaml \
  configs/GINE/finetune/ogbg-moltox21-GINE-mcl.yaml \
  configs/GINE/finetune/ogbg-molbace-GINE-mcl.yaml \
  configs/GPS/finetune/ogbg-molbbbp-GPS-mcl.yaml \
  configs/GPS/finetune/ogbg-moltox21-GPS-mcl.yaml \
  configs/GPS/finetune/ogbg-molbace-GPS-mcl.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "===== START $name ====="
  pixi run python main.py --cfg "$cfg" --repeat 3 wandb.use False
  echo "===== END $name ====="
done
