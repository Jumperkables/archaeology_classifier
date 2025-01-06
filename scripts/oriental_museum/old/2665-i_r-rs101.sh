#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --qos short
#SBATCH --job-name 2665om-i-r-rs101
#SBATCH --mem 17G
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/2665om-i-r-rs101.out
cd ../..
source venv/bin/activate
python main.py --lr=-4.035230957831288 --optimiser=Adam --model=resnet_rs_101 --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize --wandb --num_workers=0 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --encoder_freeze=0 --bsz=32 --dset_seed=2665
