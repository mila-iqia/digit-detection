#!/usr/bin/env bash

export ROOT_DIR=$HOME'/digit-detection'
export CFG=$ROOT_DIR'/config/base_config.yml'
export METADATA_FILENAME=$ROOT_DIR'/data/SVHN/train_metadata.pkl'
export DATA_DIR=$ROOT_DIR'/data/SVHN'
export RESULTS_DIR=$ROOT_DIR'/results'


# Activate your conda environment
source activate digit-detection

# Create data and results directory if needed
mkdir -p $DATA_DIR
mkdir -p $RESULTS_DIR

# Download data if needed
if [ ! -f $DATA_DIR'/train.tar.gz' ]; then

    echo "Downloading files for the training set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

# Extract data if needed
if [ ! -d $DATA_DIR/train ]; then

    echo "Extracting Files to " $DATA_DIR
    tar -xzf $DATA_DIR/train.tar.gz -C $DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

# Run training
python $ROOT_DIR'/train.py' --cfg $CFG --metadata_filename=$METADATA_FILENAME --dataset_dir=$DATA_DIR'/train' --results_dir=$RESULTS_DIR

echo "Finish training..."
