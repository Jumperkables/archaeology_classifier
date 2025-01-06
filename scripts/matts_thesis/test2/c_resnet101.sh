#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name mtt2-c_r101
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/mtt2-c_r101.out
cd ../../..
source venv/bin/activate
python main.py \
    --model resnet101 \
    --dataset matts-thesis-test2 \
    --device 0 \
    --bsz 32 \
    --lr 1e-4 \
    --vt_bsz 100 \
    --epochs 250 \
    --wandb \
    --metadata class \
    --transforms resize \
