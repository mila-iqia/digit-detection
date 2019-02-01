#!/usr/bin/env zsh

# Export commun paths
ROOT_DIR="data/milatmp1/${USER}/digit-detection"
DATA_DIR="${ROOT_DIR}/data/SVHN"
TMP_DATA_DIR="/Tmp/${USER}/digit-detection/data/SVHN"
TMP_RESULTS_DIR="/Tmp/${USER}/digit-detection/results"

# Make Tmp directories
mkdir -p $TMP_DATA_DIR
mkdir -p $TMP_RESULTS_DIR

# Check if the data already exist
if [ ! -f "${DATA_DIR}/train.tar.gz" ]; then

    echo "Downloading files for the training set!"
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

# Check if the data have been extracted from the .tar.gz file
if [ ! -d "${TMP_DATA_DIR}/train" ]; then

    echo "Extracting Files to " $TMP_DATA_DIR
    tar -xzf ${DATA_DIR}/train.tar.gz -C $TMP_DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

# Activate conda environment
source activate digit-detection

# Launch model training
cd $ROOT_DIR
python train.py --data_dir="$TMP_DATA_DIR" --result_dir="$TMP_RESULTS_DIR" "$@"

# Copy results to your own directory
echo "Copying files to local hard drive..."
cp -r $TMP_RESULTS_DIR $ROOT_DIR

# Remove the data and results from the machine used for the model training
echo "Cleaning up data..."
rm -r $TMP_DATA_DIR
rm -r $TMP_RESULTS_DIR
