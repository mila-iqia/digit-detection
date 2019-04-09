#!/bin/bash

export ROOT_DIR=$HOME'/digit-detection'
export DATA_DIR=$ROOT_DIR/data/Avenue/Humanware_v1_1553272293/
export METADATA_FILENAME=~/digit-detection/data/avenue_train_metadata_split.pkl
export CONFIG_FILE=$ROOT_DIR/config/base_config.yml

ipython -i $ROOT_DIR'/train.py' -- --dataset_dir=$DATA_DIR --metadata_filename=$METADATA_FILENAME --results_dir=$ROOT_DIR/results --cfg $CONFIG_FILE
