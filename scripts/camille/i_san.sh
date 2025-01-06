#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --qos=long-high-prio
#SBATCH --job-name=cm-i_san
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -o ../../.results/cm-i_san.out

cd ../..
source venv/bin/activate
python main.py \
    --model san \
    --dataset camille \
    --device 0 \
    --bsz 16 \
    --vt_bsz 100 \
    --lr 1e-4 \
    --epochs 250 \
    --shuffle \
    --metadata instance \
    --transforms resize \
    --wandb \
