import argparse
import time

from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import torch

import sys
sys.path.append('..')

from utils.dataloader import prepare_dataloaders


def eval_model(dataset_dir, metadata_filename, model_filename,
               batch_size, sample_size):
    '''
    Validation loop.

    Parameters
    ----------
    dataset_dir : str
        Directory with all the images.
    metadata_filename : str
        Absolute path to the metadata pickle file.
    model_filename : str
        path/filename where to save the model.
    batch_size : int
        Mini-batch size.
    sample_size : int
        Number of elements to use as sample size,
        for debugging purposes only. If -1, use all samples.

    Returns
    -------
    y_pred : ndarray
        Prediction of the model.

    '''

    seed = 1234

    print('pytorch/random seed: {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_split = 'test'

    test_loader = prepare_dataloaders(dataset_split=dataset_split,
                                      dataset_path=dataset_dir,
                                      metadata_filename=metadata_filename,
                                      batch_size=batch_size,
                                      sample_size=sample_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Load best model
    model = torch.load(model_filename, map_location=device)
    since = time.time()
    model = model.to(device)
    model = model.eval()

    print("# Testing Model ... #")
    test_correct = 0
    test_n_samples = 0
    y_true = []
    y_pred = []
    for i, batch in enumerate(tqdm(test_loader)):
        # get the inputs
        inputs, targets = batch['image'], batch['target']

        inputs = inputs.to(device)

        target_ndigits = targets[:, 0].long()
        target_ndigits = target_ndigits.to(device)

        # Forward
        outputs = model(inputs)

        # Statistics
        _, predicted = torch.max(outputs.data, 1)

        y_pred.extend(list(predicted.numpy()))
        y_true.extend(list(target_ndigits.numpy()))

        test_correct += (predicted == target_ndigits).sum().item()
        test_n_samples += target_ndigits.size(0)
        test_accuracy = test_correct / test_n_samples

    print("Confusion Matrix")
    print("===============================")
    print(confusion_matrix(y_true, y_pred, labels=range(0, 7)))
    print("===============================")
    print('\n\nTest Set Accuracy: {:.4f}'.format(test_accuracy))

    time_elapsed = time.time() - since

    print('\n\nTesting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return y_true, y_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename", type=str,
                        default='data/SVHN/test_metadata.pkl',
                        help='''metadata_filename will be the absolute
                                path to the metadata file to be used for
                                evaluation.''')

    parser.add_argument("--dataset_dir", type=str,
                        default='data/SVHN/test',
                        help='''dataset_dir will be the absolute path
                                to the directory to be used for
                                evaluation.''')

    parser.add_argument("--model_dir", type=str,
                        default='results',
                        help='''model_dir will be the absolute
                                path to the directory where the model
                                is saved.''')

    parser.add_argument("--batch_size", type=int, default=32,
                        help='Mini-batch size.')

    parser.add_argument("--sample_size", type=int, default=-1,
                        help='''Number of elements to use as sample
                                size, for debugging purposes only.
                                If -1, use all samples.''')

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    sample_size = args.sample_size

    model_filename = model_dir + '/best_model.pth'
    ground_truth_filename = model_dir + '/ground_truth.txt'
    pred_filename = model_dir + '/eval_pred.txt'

    print("\nEvaluating results ... ")
    y_true, y_pred = eval_model(
        dataset_dir, metadata_filename, model_filename,
        batch_size=batch_size, sample_size=sample_size)

    assert type(y_true) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_true.shape) == 1, "Make sure ndim=1 for y_pred"

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    assert len(y_true) == len(y_pred), "# of samples differ"

    print(ground_truth_filename)
    np.savetxt(ground_truth_filename, y_true, fmt='%.1f')

    print(pred_filename)
    np.savetxt(pred_filename, y_pred, fmt='%.1f')
