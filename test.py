import os
import time

import torch

from utils.dataloader import prepare_dataloaders


def test_model(model, test_loader, device,
               ):
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
    for i, batch in enumerate(test_loader):
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


if __name__ == "__main__":

    # CHANGE TO --args from python command
    #  results_dir = os.environ['TMP_RESULTS_DIR']
    results_dir = 'results'
    batch_size = 32

    # CHANGE TO --args from python command
    #  train_datadir = os.environ['TMP_DATA_DIR']+'/train'
    datadir = 'data/SVHN'
    test_loader = prepare_dataloaders(dataset_split='test',
                                      batch_size=batch_size,
                                      sample_size=100,
                                      datadir=datadir)

    (train_loader,
     valid_loader) = prepare_dataloaders(dataset_split='train',
                                         batch_size=batch_size,
                                         sample_size=100,
                                         datadir=datadir)

    # Define model architecture

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    #  model_filename = 'models/my_model_20181128_151805.pth'
    model_filename = 'results/my_model_20181211_144802.pth'
    #  model_filename = 'results/my_model_20181214_140235.pth'
    #  model_filename = results_dir + "/my_model"
    model = torch.load(model_filename, map_location=device)

    test_model(model,
               test_loader=test_loader,
               device=device)
