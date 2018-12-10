export DATA_DIR='/Tmp/'$USER'/data/SVHN'
export RESULTS_DIR='/Tmp/'$USER'/results/SVHN'
mkdir -p $DATA_DIR
mkdir -p RESULTS_DIR

if [ ! -d $DATA_DIR/train ]; then
    echo "Downloading files for the training set!"
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
    tar -xzf $DATA_DIR/train.tar.gz -C $DATA_DIR
    echo "Download finished, cleaning up..." 
    rm $DATA_DIR/train.tar.gz
else
    echo "Train files already present"
fi

# python train.py
cp -r $RESULTS_DIR /u/$USER 

echo "Cleaning up data..."
rm -r $DATA_DIR
rm -r $RESULTS_DIR
