#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name mtt1-i_r-rs101
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/mtt1-i_r-rs101.out
cd ../../..
source venv/bin/activate
python main.py \
    --model resnet_rs_101 \
    --dataset matts-thesis-test1 \
    --device 0 \
    --bsz 32 \
    --lr 1e-4 \
    --vt_bsz 100 \
    --epochs 250 \
    --shuffle \
    --metadata instance \
    --transforms resize \
    --optimise_hps \
    --wandb \
    #--no_preload \

