#!/bin/bash

export ROOT_DIR=$HOME'/digit-detection'
export DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train'
export TMP_DATA_DIR=$DATA_DIR #  $SCRATCH$USER'/data/SVHN'
export TMP_RESULTS_DIR=$DATA_DIR # $SCRATCH$USER'/results'
export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_metadata.pkl'

mkdir -p $TMP_DATA_DIR
mkdir -p $TMP_RESULTS_DIR

if [ ! -f $DATA_DIR'/train.tar.gz' ]; then
    
    echo "Downloading files for the training set!"
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

cp $DATA_DIR/train_metadata.pkl $TMP_DATA_DIR

if [ ! -d $TMP_DATA_DIR/train ]; then

    echo "Extracting Files to " $TMP_DATA_DIR
    cp $DATA_DIR'/train.tar.gz' $TMP_DATA_DIR
    tar -xzf $TMP_DATA_DIR'/train.tar.gz' -C $TMP_DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

python $ROOT_DIR'/train.py' --dataset_dir=$TMP_DATA_DIR --metadata_filename=$METADATA_FILENAME --results_dir=$ROOT_DIR/results --cfg $ROOT_DIR/config/base_config.yml

echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR

# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR
