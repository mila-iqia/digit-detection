
#!/bin/bash

export ROOT_DIR=$HOME'/digit-detection'
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'
export DATA_DIR=$SVHN_DIR/train
export TMP_DATA_DIR=$DATA_DIR
export TMP_RESULTS_DIR=$DATA_DIR
export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_extra_metadata_split.pkl'

mkdir -p $TMP_DATA_DIR
mkdir -p $TMP_RESULTS_DIR

s_exec python $ROOT_DIR'/train.py' --dataset_dir=$TMP_DATA_DIR --metadata_filename=$METADATA_FILENAME --results_dir=$ROOT_DIR/results --cfg $ROOT_DIR/config/base_config.yml
