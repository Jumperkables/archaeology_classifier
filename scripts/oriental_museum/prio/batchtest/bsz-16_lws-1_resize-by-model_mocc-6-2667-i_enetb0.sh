#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name bsz-16-i-resize-by-model-mocc-6-enetb0
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/bsz-16-i-resize-by-model-mocc-6-enetb0.out
cd ../../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --model=enetb0 --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --wandb --encoder_freeze=0 --bsz=16 --dset_seed=2667 --min_occ=6 --loss_weight_scaling
