#!/usr/bin/env bash

export ROOT_DIR=$HOME'/digit-detection'
export METADATA_FILENAME=$ROOT_DIR'/data/SVHN/test_metadata.pkl'
export DATA_DIR=$ROOT_DIR'/data/SVHN'
export RESULTS_DIR=$ROOT_DIR'/results'
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_01_31_10_00_52'  # 0.8903
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_07_06_28'
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_13_45_41'
# export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_14_15_15'
export MODEL_DIR=$RESULTS_DIR'/SVHN_BaselineCNN_2019_02_01_14_45_43'


# Activate your conda environment
source activate digit-detection

# Download data if needed
if [ ! -f $DATA_DIR'/test.tar.gz' ]; then

    echo "Downloading files for the test set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/test.tar.gz
fi

# Extract data if needed
if [ ! -d $DATA_DIR/test ]; then

    echo "Extracting Files to " $DATA_DIR
    tar -xzf $DATA_DIR/test.tar.gz -C $DATA_DIR
    echo "Extraction finished!"

else
    echo "Test files already present"
fi

# Run evaluation
python $ROOT_DIR'/evaluation/eval.py' --metadata_filename=$METADATA_FILENAME --dataset_dir=$DATA_DIR'/test' --model_dir=$MODEL_DIR --batch_size=32 --sample_size=-1
