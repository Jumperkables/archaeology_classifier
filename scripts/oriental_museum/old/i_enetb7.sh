#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -x gpu[0-6]
#SBATCH -t 2-00:00
#SBATCH --mem 21G
#SBATCH -p res-gpu-small
#SBATCH --job-name om-i_enetb7%j
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/om-i_enetb7%j.out
cd ../..
source venv/bin/activate

wandb agent jumperkables/archaeology/nv0nfq07
