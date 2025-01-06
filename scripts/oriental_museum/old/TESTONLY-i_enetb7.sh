#!/bin/bash
cd ../..
source venv/bin/activate
python main.py \
    --test_ckpt_path '.results/TOTEST/OM-i-enetb7/spbsz5nw/checkpoints/epoch=49-step=30649.ckpt' \
    --model enetb7 \
    --dataset oriental-museum \
    --device 0 \
    --vt_bsz 100 \
    --shuffle \
    --metadata instance \
    --transforms resize \
    --num_workers 0 \
    --dropout 0.5 \
    --fc_intermediate 2048 \
    --epochs 50 \
    --encoder_freeze 0 \
    --bsz 64 \
    --lr -6.0 \
    --optimiser Adam
