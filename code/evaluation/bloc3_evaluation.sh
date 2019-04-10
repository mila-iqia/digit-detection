#!/bin/bash

## Activate the conda environment
source activate humanware



# Compile mask rcnn
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176/lib64/:$LD_LIBRARY_PATH 

cd $HOME/$TEAM_NAME/code/maskrcnn-mila/

python setup.py build develop --user




cd $HOME

# Link datasets properly
source $HOME/b3phut_baseline/code/maskrcnn-mila/tools/avenue/link_avenue_datasets.sh

cd $HOME/b3phut_baseline/code/maskrcnn-mila

python tools/test_net.py --config-file "configs/avenue_e2e_faster_rcnn_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 5000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 4


mkdir -p $HOME/b3phut_baseline/code/temp


# Generate metadata file:

BBOX_FILE=$HOME/b3phut_baseline/code/maskrcnn-mila/inference/avenue_test_set/bbox.json

# INSTANCES_TEST_FILE=$HOME/b3phut_baseline/code/digit_detector/data/Avenue/Humanware_v1_1553272293/test_sample/instances_test_sample.json
INSTANCES_TEST_FILE=$DATA_DIR/instances_test.json

OUT_METADATA_FILE=$HOME/b3phut_baseline/code/temp/ave_test_metadata_split.pkl

python $HOME/b3phut_baseline/code/digit_detector/bbox_to_metadata.py --bbox_file=$BBOX_FILE --instances_test_file=$INSTANCES_TEST_FILE --out_metadata_file=$OUT_METADATA_FILE





# Evaluate boxes

# One path before
DATASET_DIR=$(dirname "$DATA_DIR")

RESULTS_DIR=$HOME/b3phut_baseline/code/temp

cd $HOME/b3phut_baseline/code/digit_detector/evaluation

python eval.py --metadata_filename=$OUT_METADATA_FILE --dataset_dir=$DATASET_DIR --results_dir=$RESULTS_DIR

cd $HOME/b3phut_baseline/code/
