import argparse
from pathlib import Path
import time

import numpy as np
import torch

import sys
sys.path.append('..')

from utils.dataloader import prepare_dataloaders
from trainer.trainer import batch_loop

def eval_model(dataset_dir, metadata_filename, model_filename,
               batch_size, sample_size):

    test_loader = prepare_dataloaders(dataset_dir, metadata_filename, batch_size, sample_size=-1, train=False, )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Load best model
    model = torch.load(model_filename, map_location=device)
    since = time.time()
    model = model.to(device)
    model = model.eval()

    print("# Testing Model ... #")

    print('\n\n\nIterating over testing data...')

    optimizer = None
    loss_function = None

    accuracy, total_predictions = batch_loop(test_loader, model, optimizer, loss_function, device, multiloss=True, mode='testing')

    print('\tTest Accuracy: {:.4f}'.format(accuracy))

    time_elapsed = time.time() - since

    print('\n\nTesting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    y_pred = np.asarray(total_predictions)

    return y_pred


if __name__ == "__main__":

    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename", type=str, default='data/SVHN/test_metadata_split.pkl')
    # metadata_filename will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--dataset_dir", type=str, default='data/SVHN/')
    # dataset_dir will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--results_dir", type=str, default='evaluation')
    # results_dir will be the absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    #########################################

    batch_size = 32
    sample_size = -1

    # Put your group name here
    group_name = "b2phutN_multiloss_preds"
    model_filename = "/home/jerpint/digit-detection/results/SVHN_Multiloss_VGG19_trainextra_2/best_model.pth"

    ###### DO NOT MODIFY THIS SECTION ######
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_dir, metadata_filename, model_filename,
                        batch_size=batch_size, sample_size=sample_size)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    np.savetxt(results_fname, y_pred, fmt='%.1f')
    #########################################
