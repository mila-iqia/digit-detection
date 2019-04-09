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

## Quick usage on Helios

To run the code on Helios, you can use the scripts `train_on_helios.sh`.

You can run this directly from the login node using msub:

`msub -A $GROUP_RAP -l feature=k80,nodes=1:gpus=1,walltime=2:00:00 train_on_helios.sh`

You can easily add this script to a `.pbs` file with your specific settings.

To change the data directories, you can modify the`train_on_helios.sh` script.

To change configurations during training, use the `config/base_config.yml` file.
This contains tuneable options that can be useful.

To use skopt bayesian hyperparameter optimisation use the
`config/skopt_base_config.yml` file by changing the `--cfg` flag in `train_on_helios.sh`

To add new models, create a new model file in `models/` folder  and modify
the appropriate model declaration in `trainer/trainer.py`. Currently,
VGG and Resnet are implemented.

After every run, to be able to resume an already running experience, additional information such as the output 
directory of the current experiment will be appended to the config file. These should be manually removed before new experiments.

## Basline Model

### Bloc 1
The baseline model used for the first part of the project was a VGG19
with learning rate 1e-3, momentum 0.9, SGD optimizer and batch size of 32.

### Bloc 2
The baseline model used for the second part of the project was a VGG19
with learning rate 1e-3, momentum 0.9, SGD optimizer and batch size of 32.

## Run the code interactively
For debugging purpose you might want to run your code interactively.

If you want to stop your code in a particular line you can add those
lines there: `import ipdb; ipdb.set_trace()`.
See [ipdb](https://pypi.org/project/ipdb/) for more informations.

## Data
For more information about the data used and its format, consult the `README`
in the `data/` directory.
