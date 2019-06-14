#!/bin/bash

# This script prepares files for post-processing in MATLAB with several time shifts and several cropping

max_shift=2
CKPT_FILE='/path/to/checkpoint'
DATASET_DIR='/path/to/DAVIS2016'
PWC_CKPT_FILE='/path/to/pwc_ckpt/'

for test_shift in $(seq -$max_shift $max_shift); do
        if [ ! $test_shift -eq 0 ]; then
        python3 test_generator_ensemble.py \
        --dataset=DAVIS2016 \
        --ckpt_file=$CKPT_FILE \
        --root_dir=$DATASET_DIR \
	--flow_ckpt=$PWC_CKPT_FILE \
	--test_temporal_shift=$test_shift \
	--test_partition='val' \
        --generate_visualization=True \
        --test_save_dir=/tmp/buffer_davis/davis_shift_$test_shift
        fi
done
