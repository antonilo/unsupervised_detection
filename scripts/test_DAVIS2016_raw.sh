#!/bin/bash
#
# Script to compute raw results (before post-processing)
###

# download data
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd ${SCRIPT_DIR}/..
mkdir -p downloads
# (TODO) wget DAVIS etc. to downloads/

# set params
CKPT_FILE='downloads/unsupervised_detection_models/davis_best_model/model.best'
DATASET_FILE='downloads/DAVIS/'
PWC_CKPT_FILE='downloads/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000.data-00000-of-00001'

# run test
python3 test_generator.py \
--dataset=DAVIS2016 \
--ckpt_file=$CKPT_FILE \
--flow_ckpt=$PWC_CKPT_FILE \
--test_crop=0.9 \
--test_temporal_shift=1 \
--root_dir=$DATASET_FILE \
--generate_visualization=True \
--test_save_dir=/tmp/davis_test
