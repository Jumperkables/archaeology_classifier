#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --qos short
#SBATCH --job-name om-i-augment-mocc-3-resnet_rs_101
#SBATCH --mem 17G
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/om-i-augment-mocc-3-resnet_rs_101.out
cd ../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --model=resnet_rs_101 --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=augment --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --wandb --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=3
