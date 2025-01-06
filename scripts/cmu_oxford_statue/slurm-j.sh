#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name cmuos-i_r-rs101
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/cmuos-i_r-rs101%j.out
cd ../..
source venv/bin/activate

wandb agent jumperkables/archaeology/vop7aulu
