import os
import copy
import time

import torch

from utils.dataloader import prepare_dataloaders
from models.models import BaselineCNN


def train_model(model, train_loader, valid_loader, device,
                num_epochs=10, lr=0.001, model_filename=None):

    since = time.time()
    model = model.to(device)
    train_loss_history = []
    valid_loss_history = []
    valid_accuracy_history = []
    valid_best_accuracy = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_ndigits = torch.nn.CrossEntropyLoss()

    print("# Start training #")
    for epoch in range(num_epochs):

        train_loss = 0
        train_n_iter = 0

        # Set model to train mode
        model = model.train()

        # Iterate over train data
        for i, batch in enumerate(train_loader):
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)
            target_ndigits = targets[:, 0].long()

            target_ndigits = target_ndigits.to(device)

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)

            loss = loss_ndigits(outputs, target_ndigits)

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_n_iter += 1

        valid_loss = 0
        valid_n_iter = 0
        valid_correct = 0
        valid_n_samples = 0

        # Set model to evaluate mode
        model = model.eval()

        # Iterate over valid data
        # Iterate over train data
        for i, batch in enumerate(valid_loader):
            # get the inputs
            inputs, targets = batch['image'], batch['target']

            inputs = inputs.to(device)

            target_ndigits = targets[:, 0].long()
            target_ndigits = target_ndigits.to(device)

            # Forward
            outputs = model(inputs)

            loss = loss_ndigits(outputs, target_ndigits)

            # Statistics
            valid_loss += loss.item()
            valid_n_iter += 1
            _, predicted = torch.max(outputs.data, 1)
            valid_correct += (predicted == target_ndigits).sum().item()
            valid_n_samples += target_ndigits.size(0)

        train_loss_history.append(train_loss / train_n_iter)
        valid_loss_history.append(valid_loss / valid_n_iter)
        valid_accuracy = valid_correct / valid_n_samples

        print('\nEpoch: {}/{}'.format(epoch + 1, num_epochs))
        print('\tTrain Loss: {:.4f}'.format(train_loss / train_n_iter))
        print('\tValid Loss: {:.4f}'.format(valid_loss / valid_n_iter))
        print('\tValid Accuracy: {:.4f}'.format(valid_accuracy))

        if valid_accuracy > valid_best_accuracy:
            valid_best_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)
            print('Checkpointing new model...')
        valid_accuracy_history.append(valid_accuracy)

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if model_filename:
        print('Saving model ...')
        timestr = time.strftime("_%Y%m%d_%H%M%S")
        model_filename = model_filename + timestr + '.pth'
        torch.save(best_model, model_filename)
        print('Best model saved to :', model_filename)


if __name__ == "__main__":

    # CHANGE TO --args from python command
    #  results_dir = os.environ['TMP_RESULTS_DIR']
    results_dir = 'results'
    batch_size = 32
    num_epochs = 5

    # CHANGE TO --args from python command
    #  train_datadir = os.environ['TMP_DATA_DIR']+'/train'
    datadir = 'data/SVHN'
    (train_loader,
     valid_loader) = prepare_dataloaders(dataset_split='train',
                                         batch_size=batch_size,
                                         sample_size=100,
                                         datadir=datadir)

    # Define model architecture
    baseline_cnn = BaselineCNN()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    model_filename = results_dir + "/my_model"

    train_model(baseline_cnn,
                train_loader=train_loader,
                valid_loader=valid_loader,
                num_epochs=num_epochs,
                device=device,
                model_filename=None)
