#!/bin/bash

## TODO
# cd to the shared repo with the group code
# add singularity exec commands

RESULTS_DIR='evaluation/'
SVHN_DIR='data/SVHN'

python evaluation/eval.py --SVHN_dir=$SVHN_DIR --results_dir=$RESULTS_DIR
python test.py --SVHN_dir=$SVHN_DIR --results_dir=$RESULTS_DIR

python evaluation/evaluate_stats.py --results_dir=$RESULTS_DIR
