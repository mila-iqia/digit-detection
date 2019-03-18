
#!/bin/bash

export ROOT_DIR=$HOME'/digit-detection'
export DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'

export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_extra_metadata_split.pkl'

s_exec python $ROOT_DIR'/train.py' --dataset_dir=$DATA_DIR --metadata_filename=$METADATA_FILENAME --results_dir=$ROOT_DIR/results --cfg $ROOT_DIR/config/base_config.yml
