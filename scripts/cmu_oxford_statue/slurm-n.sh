#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -x gpu[0-6]
#SBATCH -t 7-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name cmuos-i_r-rs101
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/cmuos-i_r-rs101%j.out
cd ../..
source venv/bin/activate

wandb agent jumperkables/archaeology/ra06fa71
