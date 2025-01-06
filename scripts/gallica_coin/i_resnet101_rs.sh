#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name gc-i_r-rs101
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/gc-i_r-rs101.out
cd ../..
source venv/bin/activate
python main.py \
    --model resnet_rs_101 \
    --dataset gallica_coin \
    --device 0 \
    --bsz 32 \
    --lr 1e-4 \
    --vt_bsz 100 \
    --epochs 250 \
    --shuffle \
    --wandb \
    --metadata instance \
    --transforms resize \
