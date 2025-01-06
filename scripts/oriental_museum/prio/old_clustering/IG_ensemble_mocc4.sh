#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 8G
#SBATCH -p res-gpu-small
#SBATCH --job-name IG_ensemble_mocc4
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/IG_ensemble_mocc4.out
cd ../../..
source venv/bin/activate
python misc/saliency.py \
    --model enetb0 enetb1 enetb2 enetb3 enetb4 enetb5 enetb6 enetb7 inceptionv3 inceptionv4 resnet_rs_101 \
    --test_ckpt_path "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb0-epoch=87.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb1-epoch=97.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb2-epoch=84.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb3-epoch=72.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb4-epoch=82.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb5-epoch=50.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb6-epoch=89.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_enetb7-epoch=81.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_inceptionv3-epoch=99.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_inceptionv4-epoch=98.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=4-seed2667_resnet_rs_101-epoch=99.ckpt" \
    --dataset=oriental-museum --device=0 --shuffle --metadata=instance --transforms=resize --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --encoder_freeze=0 --dset_seed=2667 --min_occ=4 --loss_weight_scaling --ensemble --method IG --save_ensemble_memory
