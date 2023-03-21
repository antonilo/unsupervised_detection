#!/bin/bash
#
# Script to compute raw results (before post-processing)
###

set -e # immediately stop after some errors.
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# parameters
DOWNLOAD_DIR="${SCRIPT_DIR}/../download"
CKPT_FILE="${DOWNLOAD_DIR}/unsupervised_detection_models/davis_best_model/model.best"
DATASET_FILE="${DOWNLOAD_DIR}/video/todaiura/todaiura_traffic.MOV"
PWC_CKPT_FILE="${DOWNLOAD_DIR}/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000.data-00000-of-00001"
RESULT_DIR="{SCRIPT_DIR}/../results/video/todaiura"


echo "[INFO] start downloading data..."
mkdir -p ${DOWNLOAD_DIR}
if [ ! -f ${CKPT_FILE}.data* ]; then
    echo "[INFO] no checkpoint file found. start downloading it."
    wget https://rpg.ifi.uzh.ch/data/unsupervised_detection_models.zip
    unzip unsupervised_detection_models.zip
    rm unsupervised_detection_models.zip
fi
if [ ! -e ${DATASET_FILE} ]; then
    echo "[INFO] no video data found. start downloading it."
    gdown "https://drive.google.com/file/d/1fTzvd1SjZmrvWoPkFgHLSXFUVmMnOESg/view?usp=sharing"
fi
if [ ! -f ${PWC_CKPT_FILE} ]; then
    echo "[INFO] no pwc checkpoint file found. start downloading it."
    gdown --folder "https://drive.google.com/drive/folders/1gtGx_6MjUQC5lZpl6-Ia718Y_0pvcYou"
fi
echo "[INFO] finished downloading."


echo "[INFO] start running a test..."
mkdir -p ${RESULT_DIR}
python3 test_video.py \
--dataset=DAVIS2016 \
--ckpt_file=$CKPT_FILE \
--flow_ckpt=$PWC_CKPT_FILE \
--test_crop=0.9 \
--test_temporal_shift=1 \
--root_dir=$DATASET_FILE \
--generate_visualization=True \
--test_save_dir=${RESULT_DIR}
echo "[INFO] finished the test."
