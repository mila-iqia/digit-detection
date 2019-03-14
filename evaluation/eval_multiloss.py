import argparse
from pathlib import Path
import time

import numpy as np
import torch

from utils.dataloader import prepare_dataloaders
from trainer.trainer import batch_loop


def eval_model(dataset_dir, metadata_filename, model_filename,
               batch_size, sample_size):

    test_loader = prepare_dataloaders(
        dataset_dir,
        metadata_filename,
        batch_size,
        sample_size=-1,
        train=False, )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Load best model
    model = torch.load(model_filename, map_location=device)
    since = time.time()
    model = model.to(device)
    model = model.eval()

    #  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #  loss_ndigits = torch.nn.CrossEntropyLoss()

    print("# Testing Model ... #")

    print('\n\n\nIterating over testing data...')

    optimizer = None
    loss_function = None

    accuracy, total_predictions = batch_loop(
        test_loader,
        model,
        optimizer,
        loss_function,
        device,
        multiloss=True,
        mode='testing')

    print('\tTest Accuracy: {:.4f}'.format(accuracy))

    time_elapsed = time.time() - since

    print('\n\nTesting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    y_pred = np.asarray(total_predictions)

    return y_pred


if __name__ == "__main__":

    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename",
                        type=str,
                        default='data/SVHN/test_metadata_split.pkl',
                        help='''absolute path to the directory to be used
                                for evaluation''')

    parser.add_argument("--dataset_dir",
                        type=str,
                        default='data/SVHN/',
                        help='''absolute path to a directory where the output
                                of your inference will be saved''')

    parser.add_argument("--results_dir",
                        type=str,
                        default='evaluation',
                        help='''absolute path to a directory where the output
                                of your inference will be saved.''')

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    #########################################

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--model_filename", type=str,
                        default='/home/jerpint/digit-detection/results/SVHN_TestMultilossJP/best_model.pth')

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    batch_size = args.batch_size
    sample_size = args.sample_size
    model_filename = args.model_filename

    # Put your group name here
    group_name = "b1phutN_multiloss_test"

    #  metadata_filename = Path(SVHN_dir) / 'test_metadata.pkl'
    #  dataset_path = Path(SVHN_dir) / 'test'

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
