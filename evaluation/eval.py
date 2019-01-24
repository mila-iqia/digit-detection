import argparse
from pathlib import Path
import time

from tqdm import tqdm
import numpy as np
import torch

from utils.dataloader import prepare_dataloaders


def eval_model(dataset_dir, metadata_filename, model_filename,
               batch_size, sample_size):

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

    #  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #  loss_ndigits = torch.nn.CrossEntropyLoss()

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

        #  loss = loss_ndigits(outputs, target_ndigits)

        # Statistics
        #  test_loss += loss.item()
        #  test_n_iter += 1
        _, predicted = torch.max(outputs.data, 1)

        y_pred.extend(list(predicted.numpy()))
        y_true.extend(list(target_ndigits.numpy()))

        test_correct += (predicted == target_ndigits).sum().item()
        test_n_samples += target_ndigits.size(0)
        test_accuracy = test_correct / test_n_samples

    from sklearn.metrics import confusion_matrix

    print("Confusion Matrix")
    print("===============================")
    print(confusion_matrix(y_true, y_pred, labels=range(0, 7)))
    print("===============================")
    print('\n\nTest Set Accuracy: {:.4f}'.format(test_accuracy))

    time_elapsed = time.time() - since

    print('\n\nTesting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    y_pred = np.asarray(y_pred)

    return y_pred

    #  results_gt_fname = Path(results_dir) / 'test_gt_output.txt'
    #  np.savetxt(results_gt_fname, y_true, fmt='%.1f')


if __name__ == "__main__":

    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename", type=str, default='data/SVHN/test_metadata.pkl')
    # metadata_filename will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--dataset_dir", type=str, default='data/SVHN/test')
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

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--model_filename", type=str,
                        default='results/my_model_20181211_144802.pth')

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    batch_size = args.batch_size
    sample_size = args.sample_size
    model_filename = args.model_filename

    # Put your group name here
    group_name = "b1phutN"

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
