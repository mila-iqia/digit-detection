#!/bin/bash

export DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN'

# Download data
if [ ! -f $DATA_DIR'/train.tar.gz' ]; then

    echo "Downloading files for the training set to "$DATA_DIR
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi
