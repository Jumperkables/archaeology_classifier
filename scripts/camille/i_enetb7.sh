#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name cm-i_enetb7
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/cm-i_enetb7.out
cd ../..
source venv/bin/activate
python main.py \
    --model enetb7 \
    --dataset camille \
    --device 1 \
    --bsz 32 \
    --lr 1e-4 \
    --vt_bsz 100 \
    --epochs 250 \
    --shuffle \
    --metadata instance \
    --transforms resize \
    #--wandb \
