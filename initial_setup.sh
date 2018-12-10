DATA_DIR='/Tmp/'$USER'/data/SVHN'
mkdir -p $DATA_DIR

if [ ! -d $DATA_DIR/train ]; then
    echo "Downloading files for the training set!"
    wget -P $DATA_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
    tar -xzf $DATA_DIR/train.tar.gz -C $DATA_DIR
    echo "Download finished, cleaning up..." 
    rm $DATA_DIR/train.tar.gz
else
    echo "Train files already present"
fi
