# Humanware Project

This repo contains the code necessary for the Humanware project.

## Dependencies

We recommend you use [conda](https://conda.io/docs/) to manage your environments. You can start with a preconfigured environment by using the provided `environment.yml` file:

`conda env create -f environment.yml`

This will create a new conda environment with the libraries used for this project (i.e. pytorch, torchvision, etc.) called `digit-detection`. For more information on how to manage different environments, consult the [conda documentation](https://conda.io/docs/user-guide/tasks/manage-environments.html) 

## Run Locally

### Get the data

To get the data, run the `local_setup.sh` script which will download the data to the appropriate directories. Make sure the file is executable:

```
chmod +x local_setup.sh
./local_setup.sh
```

To run your code locally, activate your conda environment. Make sure you have already created the conda environment from the `environment.yml` file. See the `Dependencies` section for more info.

`conda activate digit-detection`

then run the provided `train.py` file:

`python train.py`

## Run on the cluster

To run the code on the cluster, use the `train_on_slurm.sh` file. That script will:
- Copy your data to the proper directories
- Execute training
- Save models
- Clean up your data

## Data

For more information about the data used and its format, consult the `README` in the `data/` directory.

## Testing and model evaluation

To evaluate your model, use the provided `test.py` file.
