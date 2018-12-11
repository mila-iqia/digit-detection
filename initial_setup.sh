#!/bin/bash

export DATA_DIR='data/milatmp1/'$USER 
export TMP_DATA_DIR='/Tmp/'$USER'/data/SVHN'
export TMP_RESULTS_DIR='/Tmp/'$USER'/results'
export ROOT_DIR=$HOME'/digit-detection'

mkdir -p $TMP_DATA_DIR
mkdir -p $TMP_RESULTS_DIR

if [ ! -f $DATA_DIR'/train.tar.gz' ]; then
    
    echo "Downloading files for the training set!"
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

if [ ! -d $TMP_DATA_DIR/train ]; then

    echo "Extracting Files to " $TMP_DATA_DIR
    cp $DATA_DIR'/train.tar.gz' $TMP_DATA_DIR
    tar -xzf $TMP_DATA_DIR/train.tar.gz -C $TMP_DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

conda env create -f $ROOT_DIR'/environment.yml'
source activate digit-detection
python train.py

echo "Copying files to local hard drive..."
cp -r $TMP_RESULTS_DIR $ROOT_DIR

echo "Cleaning up data..."
rm -r $TMP_DATA_DIR
rm -r $TMP_RESULTS_DIR
