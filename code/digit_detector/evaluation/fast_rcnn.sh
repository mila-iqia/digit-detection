#!/bin/bash


# This will contain the avenue test set and .json file
export HUMANWARE_DATA_DIR = '/rap/jvb-000-aa/COURS2019/assistants/Humwanware_v1_15535195/test/'

### COMPLETE THIS ACCORDINGLY
cd PROJECT_PATH/code/maskrcnn-code
###


### DO NOT MODIFY
# Activate conda environment
source activate humanware
python setup.py build develop --user # Compile code
### DO NOT MODIFY




### COMPLETE THIS ACCORDINGLY
### Run your execution

python tools/test_net.py # Add path to your configs and all other parameters

