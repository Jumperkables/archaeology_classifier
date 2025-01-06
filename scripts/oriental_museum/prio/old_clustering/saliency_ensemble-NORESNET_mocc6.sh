#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name saliency_ensemble-NORESNET_mocc6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/saliency_ensemble-NORESNET_mocc6.out
cd ../../..
source venv/bin/activate
python misc/saliency.py \
    --model enetb0 enetb1 enetb2 enetb3 enetb4 enetb5 enetb6 enetb7 inceptionv3 inceptionv4 \
    --test_ckpt_path "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb0-epoch=75.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb1-epoch=55.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb2-epoch=72.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb3-epoch=97.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb4-epoch=85.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb5-epoch=89.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb6-epoch=84.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_enetb7-epoch=92.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_inceptionv3-epoch=97.ckpt" "../.results/FIXED_lws1-resize-oriental-museum-['instance']-minocc=6-seed2667_inceptionv4-epoch=93.ckpt" \
    --dataset=oriental-museum --device=0 --shuffle --metadata=instance --transforms=resize --num_workers=4 --dropout=0.5 --fc_intermediate=2048 --encoder_freeze=0 --dset_seed=2667 --min_occ=6 --loss_weight_scaling --ensemble --method saliency
