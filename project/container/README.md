Containers are virtualizing tools that enforce a stable development environment throughout devices to help with, among other things, the good reproducibility of science. You can read an overview of what is a container [here](https://www.docker.com/resources/what-container).

The two most well-known container platforms are [Docker](https://www.docker.com/) and [Singularity](https://www.sylabs.io/guides/3.0/user-guide/). In this class, we will use Singularity version 3.

The Singularity image that you will need to use is stored on the Compute Canada cluster Helios at the following path `/rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg`.

## Run
To be able to run this image on Helios you need to first run the following command to load singularity.

`source /rap/jvb-000-aa/COURS2019/etudiants/common.env`

And then append it at the end of your `~/.bashrc` using 

`echo 'source /rap/jvb-000-aa/COURS2019/etudiants/common.env' >> ~/.bashrc`

When singularity is properly loaded, you simply have to run `singularity shell --nv /rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg` to load all the dependencies you should need and you are ready to run some code.

## Build
You should not need to re-build an image from scratch but if you are interested it goes as follows. Note that you need admin right to be able to build images.

To build the ift6759.simg image you first need to download the [recipe we used](https://github.com/mila-udem/ift6759/blob/master/container/build_ift6759_img.sh) and [version 10 of cuDNN](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.2/prod/10.0_20181213/cudnn-10.0-linux-x64-v7.4.2.24.tgz) to your home directory.

Then edit `build_ift6759_img.sh` to set the proper path to your home directory.

And finally run `sudo singularity build ift6759.simg build_ift6759_img.sh` and wait for a good 30 minutes.
