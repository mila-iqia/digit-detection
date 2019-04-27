# Example Singularity recipe for pytorch on helios
#
# Build with:
# singularity build --force pytorch.simg Singularity

Bootstrap: docker
From: ubuntu:16.04

#IncludeCmd: yes

%runscript
    echo "A PyTorch Singularity container"
    exec echo "Enter the container using 'singularity shell --nv <container>'"

%files
    # CUDNN archive must be on local host
    # Download https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.2/prod/10.0_20181213/cudnn-10.0-linux-x64-v7.4.2.24.tgz to your home dir
    /home/YOUR_LOCAL_HOME_CHANGE_THIS/cudnn-10.0-linux-x64-v7.4.2.24.tgz /cudnn-10.0-linux-x64-v7.4.2.24.tgz

%environment
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/miniconda/bin:${PATH}
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    export PATH LD_LIBRARY_PATH

%labels
#    AUTHOR MILA staff

%post
    # Basic utilies
    apt-get update
    apt-get install -y wget gnupg2 curl ca-certificates libopenblas-dev ninja-build vim emacs nano htop less
    DEBIAN_FRONTEND=noninteractive apt-get install keyboard-configuration -y

    # Install CUDA from network deb package
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
    dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    apt-get update
    apt-get install -y cuda
    ln -s cuda /usr/local/cuda

    # Install CUDNN from file on local host
    tar -xvzf cudnn-10.0-linux-x64-v7.4.2.24.tgz
    cp cuda/include/cudnn.h /usr/local/cuda/include
    cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    chmod a+r /usr/local/cuda/include/cudnn.h

    # Setup conda python
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ./Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
    export PATH=/miniconda/bin:$PATH

    # Install pytorch with conda
    conda install python=3.6 pytorch=1.0.0 torchvision=0.2.1 cuda100 -c pytorch
    conda install ipython scikit-learn holoviews seaborn tqdm packaging appdirs git opencv scikit-image joblib
    conda install -c conda-forge tensorboardx multicore-tsne
    pip install gpustat easydict comet_ml
