#!/bin/bash

# Script to compute raw results (before post-processing)

CKPT_FILE='/path/to/davis2016_ckpt/'
DATASET_FILE='/path/to/DAVIS_2016/'
PWC_CKPT_FILE='/path/to/pwc_ckpt/'

python3 test_generator.py \
--dataset=DAVIS2016 \
--ckpt_file=$CKPT_FILE \
--flow_ckpt=$PWC_CKPT_FILE \
--test_crop=0.9 \
--test_temporal_shift=1 \
--root_dir=$DATASET_FILE \
--generate_visualization=True \
--test_save_dir=/tmp/davis_test
