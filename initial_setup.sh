export TMP_DATA_DIR='/Tmp/'$USER'/data/SVHN'
export TMP_RESULTS_DIR='/Tmp/'$USER'/results'
export ROOT_DIR=$HOME'/digit-detection'

mkdir -p $TMP_DATA_DIR
mkdir -p $TMP_RESULTS_DIR


if [ ! -d $TMP_DATA_DIR/train ]; then
    echo "Downloading files for the training set!"
    wget -P $TMP_DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
    tar -xzf $TMP_DATA_DIR/train.tar.gz -C $TMP_DATA_DIR
    echo "Download finished, cleaning up..." 
    rm $TMP_DATA_DIR/train.tar.gz
else
    echo "Train files already present"
fi
. /home/jerpint/miniconda3/etc/profile.d/conda.sh
conda env create -f $ROOT_DIR'environment.yml'
conda activate digit-detection
python train.py

echo "Copying files to local hard drive..."
echo $TMP_RESULTS_DIR
cp -r $TMP_RESULTS_DIR $ROOT_DIR

echo "Cleaning up data..."
rm -r $TMP_DATA_DIR
rm -r $TMP_RESULTS_DIR
