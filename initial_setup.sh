if [ ! -f data/SVHN/test.tar.gz ]; then
    echo "Downloading Files!"
    wget -P ./data/SVHN/ http://ufldl.stanford.edu/housenumbers/test.tar.gz
    tar -xzf ./data/SVHN/test.tar.gz -C ./data/SVHN/
else
    echo "Files already present"
fi

