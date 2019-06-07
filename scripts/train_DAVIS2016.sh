#!/bin/bash

# Script to Train a model on the DAVIS 2016 dataset (https://davischallenge.org/index.html)

python3 train.py 
--flow_normalizer=80.0 \
--epsilon=75.0 \
--max_temporal_len=2 \
--train_crop=0.6 \
--test_crop=0.9 \
--iters_rec=1 \
--iters_gen=3 \
--dataset=DAVIS2016 \
--root_dir='/path/to/DAVIS_2016/' \
--flow_ckpt='/path/to/PWCNet/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000' \
--recover_ckpt='/path/to/pretrained_recover/model-175'
--test_temporal_shift=1 \
--checkpoint_dir=/tmp/tests
