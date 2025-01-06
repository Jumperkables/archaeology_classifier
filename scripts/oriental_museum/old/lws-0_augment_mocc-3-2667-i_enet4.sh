#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name om-i-augment-mocc-3-enetb4
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/om-i-augment-mocc-3-enetb4.out
cd ../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --model=enetb4 --dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=augment --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --wandb --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=3
