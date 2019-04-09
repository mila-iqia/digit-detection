#!/bin/bash

## Activate the conda environment
source activate humanware



# generate metadata file:

BBOX_FILE=$HOME/digit-detection/code/digit_detector/results/inference/avenue_test_set/bbox.json

INSTANCES_TEST_FILE=$HOME/digit-detection/code/digit_detector/data/Avenue/Humanware_v1_1553272293/test_sample/instances_test_sample.json

OUT_METADATA_FILE=$HOME/digit-detection/code/temp/ave_test_metadata_split.pkl

python $HOME/digit-detection/code/digit_detector/bbox_to_metadata.py --bbox_file=$BBOX_FILE --instances_test_file=$INSTANCES_TEST_FILE --out_metadata_file=$OUT_METADATA_FILE





### FILL OUT THE REST OF THE FILE ###

DATASET_DIR=$HOME/digit-detection/code/digit_detector/data/Avenue/Humanware_v1_1553272293/

RESULTS_DIR=$HOME/digit-detection/code/temp

cd $HOME/digit-detection/code/digit_detector/evaluation
python eval.py --metadata_filename=$OUT_METADATA_FILE --dataset_dir=$DATASET_DIR --results_dir=$RESULTS_DIR

cd $HOME/digit-detection/code/

# You should build maskrcnn-bechnmark
# The final results should be saved to $RESULTS_DIR
# Refer to evaluation_instructions.md for more information
