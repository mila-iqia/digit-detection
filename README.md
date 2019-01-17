# Door Number Detection Project

This repository contains the code necessary for the door number detection
project.

The goal of the project is to help blind persons to find their way around by
making sure they are at the right house when they want for example visit a
friend or a family member, go to a specific store, etc.

In developing this project we must keep in mind the different constraints of
this application notably for the selection and development of the models we
will use like the execution time, online vs. offline, the memory usage (in the
case of a mobile application), etc.

## Dependencies

### Use conda to manage your environments

Verify that conda is installed and running on your system by typing
`conda --version`. Conda displays the number of the version that you have
installed. EXAMPLE: `conda 4.5.12`

We recommend that you always keep conda updated to the latest version.
To update conda type: `conda update conda`.

Some resources that can be useful:
[Install conda](https://conda.io/docs/user-guide/install/index.html)
[Getting started with conda](https://conda.io/docs/user-guide/getting-started.html)

### Create your environment

We strongly recommend that you start with our preconfigured
environment by using the provided `environment.yml` file by running in your
terminal from the root directory `conda env create -f environment.yml`.

Note:
- To see a list of all your environment type `conda env list`
- To see the list of the packages in your new environment type
`conda list -n digit-detection`
- To remove an environment type `conda env remove --name myenv`

Resource that can be useful:
[manage environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)

### Activate and deactivate your environment
To activate the created environment, type `source activate digit-detection`.
To deactivate your environment, type `source deactivate`.

## Run the code interactively

To run your code, activate your conda environment in the root directory:
`source activate digit-detection`

Note: Before running the code you need to download the data. Run:
- `wget -P data/SVHN/ http://ufldl.stanford.edu/housenumbers/train.tar.gz`
- `tar -xzf data/SVHN/train.tar.gz data/SVHN/`

To train the model run the provided `train.py` file.
Then, to evaluate your model, use the provided `test.py` file.

## Run the code

To run the code, use the `train_on_slurm.sh` file.
That script will:
- Copy your data to the proper directories
- Execute training
- Save models
- Clean up your data

Note: You might want to change the path for storing data and results.

## Data
For more information about the data used and its format, consult the `README`
in the `data/` directory.
