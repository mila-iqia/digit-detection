# Humanware Project

This repo contains the code necessary for the Humanware project.

## Dependencies

Use conda to manage your environments. You can start with a preconfigured environment by using the provided `environment.yml` file.

## Run Locally

To run your code locally, activate your conda environment, then run the provided `train.py` file.

## Cluster

To run the code on the cluster, use the `train_on_slurm.sh` file. That script will:
- Copy your data to the proper directories
- Execute training
- Save models
- Clean up your data

For more information about the data used and its format, consult the `README` in the `data/` directory.

To evaluate your model, use the provided `test.py` file.
