import argparse
from pathlib import Path

import numpy as np
import torch


def eval_model(model_filename, metadata_filename):

    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.

    '''
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load your best model
    if model_filename:
        model_filename = Path(model_filename)
        print("\nLoading model from", model_filename.absolute())
        model = torch.load(model_filename, map_location=device)

    if model:

        # # # # # # # # # # # #
        # Add inference here  #
        # # # # # # # # # # # #
        pass

    else:

        print("\nYou did not specify a model, generating dummy data instead!")
        n_classes = 5
        y_pred = np.random.randint(0, n_classes, (100))

    return y_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--SVHN_dir", type=str, default='')
    # data_dir will be the absolute path to the SVHN directory to be used for
    # evaluation. It will be set be the evaluator. It will consist of the same
    # structure as the SVHN directory on the shared drive. You can assume that
    # the test/ folder will already be extracted.

    parser.add_argument("--results_dir", type=str, default='')
    # results_dir will be an absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    SVHN_dir = args.SVHN_dir
    results_dir = args.results_dir

    # Put your group name here
    group_name = "b1phutN"

    metadata_filename = Path(SVHN_dir) / 'test_metadata.pkl'
    dataset_path = Path(SVHN_dir) / 'test'

    model_filename = None
    # model_filename should be the absolute path on shared disk to your
    # best model. You need to ensure that they are available to evaluators on
    # Helios.

    print("\nEvaluating results ... ")
    y_pred = eval_model(model_filename, metadata_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    np.savetxt(results_fname, y_pred, fmt='%.1f')
