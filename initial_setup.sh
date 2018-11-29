if [ ! -d data/SVHN/train]; then
    echo "Downloading files for the training set!"
    wget -P ./data/SVHN/ http://ufldl.stanford.edu/housenumbers/train.tar.gz
    tar -xzf ./data/SVHN/train.tar.gz -C ./data/SVHN/
    echo "Download finished, cleaning up..." 
    rm ./data/SVHN/train.tar.gz
else
    echo "Train files already present"
fi

