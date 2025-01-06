#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 22G
#SBATCH -p res-gpu-small
#SBATCH --job-name cm-i_r-rs101
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/cm-i_r-rs101%j.out
cd ../..
source venv/bin/activate

wandb agent jumperkables/archaeology/k7bstdh4
