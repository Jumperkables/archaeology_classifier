#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -x gpu[0-6]
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name cm-i_san
#SBATCH --gres gpu:1
#SBATCH -o ../../.results/cm-i_san%j.out
cd ../..
source venv/bin/activate

wandb agent jumperkables/archaeology/1lhycwt4
