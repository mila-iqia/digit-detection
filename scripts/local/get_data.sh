#!/usr/bin/env bash

export DATA_DIR=$HOME'/digit-detection/data/SVHN'

# Make sure the data directory exist
mkdir -p $DATA_DIR


# Download data
if [ ! -f $DATA_DIR'/train.tar.gz' ]; then

    echo "Downloading files for the training set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

if [ ! -f $DATA_DIR'/extra.tar.gz' ]; then

    echo "Downloading files for the extra set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/extra.tar.gz
fi

if [ ! -f $DATA_DIR'/test.tar.gz' ]; then

    echo "Downloading files for the test set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/test.tar.gz
fi


# Extract data
if [ ! -d $DATA_DIR/train ]; then

    echo "Extracting train files to " $DATA_DIR
    tar -xzf $DATA_DIR/train.tar.gz -C $DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

if [ ! -d $DATA_DIR/extra ]; then

    echo "Extracting extra files to " $DATA_DIR
    tar -xzf $DATA_DIR/extra.tar.gz -C $DATA_DIR
    echo "Extraction finished!"

else
    echo "Extra files already present"
fi

if [ ! -d $DATA_DIR/test ]; then

    echo "Extracting test files to " $DATA_DIR
    tar -xzf $DATA_DIR/test.tar.gz -C $DATA_DIR
    echo "Extraction finished!"

else
    echo "Test files already present"
fi
