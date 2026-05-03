#!/bin/bash
# Run ALL 7 OTFormer molmcl finetune datasets sequentially
set -e
cd /home/liar/ml/PAGraphomer

LOG="/tmp/otformer_molmcl_all.log"
echo "===== OTFormer MolMCL Finetune - ALL 7 datasets =====" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

for ds in bbbp clintox muv hiv bace tox21 sider; do
    name="ogbg-mol${ds}"
    cfg="configs/OTFormer/finetune/${name}-OTFormer-finetune-molmcl.yaml"
    out="results/finetune/${name}-OTFormer-finetune-molmcl"
    
    echo "" | tee -a "$LOG"
    echo "===== [$(date)] Starting ${name} =====" | tee -a "$LOG"
    rm -rf "$out"
    
    pixi run python main.py --cfg "$cfg" wandb.use False 2>&1 | tee -a "$LOG"
    
    echo "===== [$(date)] Finished ${name} =====" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "===== ALL DONE @ $(date) =====" | tee -a "$LOG"
