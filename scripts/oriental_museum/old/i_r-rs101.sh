#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name om-i_r-rs101%j
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/om-i_r-rs101%j.out
cd ../..
source venv/bin/activate

wandb agent jumperkables/archaeology/ufa0z94b
