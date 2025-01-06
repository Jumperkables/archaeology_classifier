#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 8G
#SBATCH -p res-gpu-small
#SBATCH --job-name conf-matrix_best_mocc6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/conf-matrix_best_mocc6.out
cd ../../..
source venv/bin/activate
python main.py --lr=-4.8 --optimiser=Adam --conf_matrix \
    --model enetb2 \
    --test_ckpt_path ".results/bsz32-lws1-resize-by-model-oriental-museum-['instance']-minocc=6-seed2667_enetb2-epoch=80.ckpt" \
    --dataset=oriental-museum --device=1 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize-by-model --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --epochs=100 --encoder_freeze=0 --bsz=32 --dset_seed=2667 --min_occ=6 --loss_weight_scaling
#--dataset=oriental-museum --device=0 --vt_bsz=100 --shuffle --metadata=instance --transforms=resize-by-model --fc_intermediate=2048 --encoder_freeze=0 --dset_seed=2667 --min_occ=6 --loss_weight_scaling
