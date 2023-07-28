#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python train.py \
-split train \
-dataset Custom \
-root data \
-lr 0.0001 \
-epoch 200 \
-batch_size 16 \
-n_frames 70 \
-n_filters 80 \
-n_joints 48 \
-feat_dims 1128 \
-num_workers 1 \
-is_balanced False \
-weighted False \
-transform True \
