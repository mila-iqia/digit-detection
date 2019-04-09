# Evaluation instructions for Humanware Bloc 3

Due to the pipeline structure of bloc 3 and its dependencies, please follow this guide to ensure that your submission goes smoothly. Should you have any questions about the submission guide, please contact [Jeremy](jeremy.pinto@mila.quebec) and / or open issues on the course [github](https://github.com/mila-iqia/ift6759/issues).

## Submissions

Your code will be executed using the script `run_evaluation.sh`. You can find it on the course's [github](https://github.com/mila-iqia/ift6759/tree/master/projects/humanware/evaluation). You are responsible to ensure that once the script is executed, the proper output is saved to the proper paths, which are specified in `run_evaluation.sh`. Take a close look at `run_evaluation.sh`. It will call `s_exec` on `bloc3_evaluation.sh`. This means that all subsequent code will be running from inside the singularity containers.

It is your responsibility to fill out `bloc3_evaluation.sh`. You can use the `eval.py` file for inspiration that was used for blocs 1 and 2.

## Environment Variables

Four exported variables will be made available to you:

* `$TEAM_NAME` - Your unique team name identifier, i.e. b3phutN, where N is your team number

* `$PROJECT_PATH` - It will be the full path to your team's submission, i.e. :

`export PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/submissions/'$TEAM_NAME`

* `$RESULTS_DIR` - It will be the full path to the folder where your final results should be saved.

* `$DATA_DIR` - It will contain the path to the `test` folder. The test set will contain all images as well as a file, `instances_test.json`, structured identically to the `instances_train.json` and `instances_valid.json`.

You will have a sample test set made available to test your submissions.

## Suggested steps to ensure a proper submission

1. Compile mask-rcnn benchmark. Don't forget to `source activate humanware`!

2. Run the trained faster-rcnn model on the test set and extract all information needed for the bounding boxes. The `maskrcnn-benchmark/tools/test_net.py` will be useful in this case.

3. Run the bloc 2 fine-tuned model on the test set and output a results file *WITH YOUR TEAMNAME PROPERLY IDENTIFIED* to `$RESULTS_DIR`. For step 3, it is recommended you use eval.py similarly to how it was used in bloc 2. If all steps are followed correctly, your results file will be written to `$RESULTS_DIR/b3phutN_preds.txt`, where `N` is your own team name.

The structure of `b3phutN_preds.txt` should be identical to the one submitted for blocs 1 and 2, i.e. a numpy array saved as a .txt file. Refer to `eval.py` from bloc 1 and 2 for further instructions.


## Additional notes

Your entire folder will be copied to the evaluator's $HOME before execution, so you should not worry about read/write permissions throughout your submission.

You are responsible for making sure your code runs. If you have any doubts, contact the instructors.

All evaluation scripts should be placed in $PROJECT_PATH/code/evaluation/. Be sure to give them the proper execute permission (use `chmod +x /file/to/be/executed`)
