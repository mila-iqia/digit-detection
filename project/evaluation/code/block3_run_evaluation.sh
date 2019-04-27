#!/bin/bash

# The expectation is that running this script on helios using msub will do the end-to-end evaluation

export TEAM_NAME=b3pdndN

# PROJECT_PATH will designate the full path to the folder containing your code and repo, i.e.
export PROJECT_PATH=/rap/jvb-000-aa/COURS2019/etudiants/submissions/$TEAM_NAME

# RESULTS_DIR will contain the full path to Where the final evaluation results will be saved. In this case, the final evaluation result is $TEAM_NAME
export RESULTS_DIR=/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/digit-detection/evaluation

# Where the test data is located
export DATA_DIR=/rap/jvb-000-aa/COURS2019/assistants/digit-detection/test

# Your project will be copied to the evaluator's home to ensure proper write permission, so you don't have to worry about those
cp -r $PROJECT_PATH $HOME


# The code will be launched using s_exec, meaning that every subsequent script can be assumed to already be inside a singularity container.

chmod +x $HOME/$TEAM_NAME/code/evaluation/bloc3_evaluation.sh # Make code executable

cd $HOME/$TEAM_NAME
s_exec code/evaluation/bloc3_evaluation.sh
