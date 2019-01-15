#!/bin/bash

export DATA_DIR=$HOME'/digit-detection/data/SVHN'

# Download data
if [ ! -f $DATA_DIR'/train.tar.gz' ]; then
    
    echo "Downloading files for the training set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

# Extract data
if [ ! -d $DATA_DIR/train ]; then

    echo "Extracting Files to " $DATA_DIR
    tar -xzf $DATA_DIR/train.tar.gz -C $DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi
