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

To run the code on Helios, you can use the scripts in `scrips/helios/train_on_helios.sh`. 

You can run this directly from the login node using msub: 

`msub -A $GROUP_RAP -l feature=k80,nodes=1:gpus=1,walltime=2:00:00 scripts/helios/train_on_helios.sh`

You can easily add this script to a `.pbs` file with your specific settings.

To change the data directories, you can modify the `train_on_helios.sh` script. To change configurations during training, use the `config/base_config.yml` file. This contains tuneable options that can be useful.

## Dependencies (For local setups)

### Use conda to manage your environments

Verify that conda is installed and running on your system by typing
`conda --version` from your terminal. Conda displays the number of the
version that you have installed. EXAMPLE: `conda 4.5.12`

We recommend that you always keep conda updated to the latest version.
To update conda type: `conda update conda`.

Some resources that can be useful:
[Install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
[Getting started with conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)

### Create your environment

We strongly recommend that you start with our preconfigured
environment by using the provided `envs/environment.yml` file by running in your
terminal from the root directory `conda env create -f envs/environment.yml`.
If you have a mac use the `envs/mac-environment.yml` file. You can also use the provided `Pipfile` if you prefer using `pipenv`.

Note:
- To activate the created environment, type
`source activate digit-detection`.
- To deactivate your environment, type `source deactivate`.
- To see a list of all your environment type `conda env list`.
- To see the list of the packages in your new environment type
`conda list -n digit-detection`.
- If a package is missing, you can add it via `conda install package`
or `pip install package` after you activate your environment.
- To remove an environment type `conda env remove --name myenv`.

Resource that can be useful:
[manage environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Run the code (from your terminal)
First add in your `~/.bashrc` the python path to the digit-detection folder:
`export PYTHONPATH="${PYTHONPATH}:/Users/johndoe/digit-detection"`

To get the data run: `scripts/local/get_data.sh`

To train the model run: `scripts/local/train.sh`

To evaluate your model run: `scripts/local/eval.sh`

Notes:
- You might want to change the path for storing data and results in
the `.sh` files.
- If the .sh files does not appear to be executable files do
`chmod +x file_name.sh` and it should work.
- You can use `command &> output.txt` to save both the standard output
and standard error display in your terminal. By doing this way,
stream will be redirected to the file only, nothing will be visible in
the terminal. If the file already exists, it gets overwritten.
- You can use `command |& tee output.txt` to save both the standard output
and standard error display in your terminal. By doing this way,
streams will be copied to the file while still being visible in the
terminal. If the file already exists, it gets overwritten.
- You can run `nohup command > output.txt &` and all output, including
any error messages, will be written to the file `output.txt`. If
`command` is running when you log out or close the terminal,
`command` will not terminate.
- You can run `ps aux | grep bash` to see bash process that are running.

## Run the code interactively
For debugging purpose you might want to run your code interactively.

If you want to stop your code in a particular line you can add those
lines there: `import ipdb; ipdb.set_trace()`.
See [ipdb](https://pypi.org/project/ipdb/) for more informations.

## Data
For more information about the data used and its format, consult the `README`
in the `data/` directory.
