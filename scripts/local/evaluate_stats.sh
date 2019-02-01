#!/usr/bin/env bash

export ROOT_DIR=$HOME'/digit-detection'
export METADATA_FILENAME=$ROOT_DIR'/data/SVHN/train_metadata.pkl'
export DATA_DIR=$ROOT_DIR'/data/SVHN'
export RESULTS_DIR=$ROOT_DIR'/results'
export MODEL_DIR1=$RESULTS_DIR'/SVHN_BaselineCNN_2019_01_31_10_00_52'
export MODEL_DIR2=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_07_06_28'
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_13_45_41'
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_14_15_15'
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_14_45_43'


# Activate your conda environment
source activate digit-detection

# Run statistical evaluation
python $ROOT_DIR'/evaluation/evaluate_stats.py' --y_true $MODEL_DIR1'/ground_truth.txt' --y_pred1 $MODEL_DIR1'/eval_pred.txt' --y_pred2 $MODEL_DIR2'/eval_pred.txt'
