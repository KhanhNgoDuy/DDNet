#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python train.py \
-split train \
-dataset Custom \
-root data \
-lr 0.001 \
-epoch 200 \
-batch_size 16 \
-n_frames 60 \
-n_filters 64 \
-feat_dims 861 \
-num_workers 2 \
-is_balanced True \
-weighted False \
#-load_model b_70_128/_120.pth \
