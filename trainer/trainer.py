from __future__ import print_function

import copy
import time

import torch
from tqdm import tqdm

from utils.config import cfg


def train_model(model, train_loader, valid_loader, device,
                num_epochs=cfg.TRAIN.NUM_EPOCHS, lr=cfg.TRAIN.LR,
                output_dir=None):
    '''
    Training loop.

    Parameters
    ----------
    model : obj
        The model.
    train_loader : obj
        The train data loader.
    valid_loader : obj
        The validation data loader.
    device : str
        The type of device to use ('cpu' or 'gpu').
    num_eopchs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    output_dir : str
        path to the directory where to save the model.

    '''

    since = time.time()
    model = model.to(device)
    train_loss_history = []
    valid_loss_history = []
    valid_accuracy_history = []
    valid_best_accuracy = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOM)
    loss_ndigits = torch.nn.CrossEntropyLoss()

    print("# Start training #")
    for epoch in range(num_epochs):

        train_loss = 0
        train_n_iter = 0

        # Set model to train mode
        model = model.train()

        # Iterate over train data
        print("\n\n\nIterating over training data...")
        for i, batch in enumerate(tqdm(train_loader)):
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
        print("Iterating over validation data...")
        for i, batch in enumerate(tqdm(valid_loader)):
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
            model_filename = output_dir + '/checkpoint.pth'
            torch.save(model, model_filename)
        valid_accuracy_history.append(valid_accuracy)

    time_elapsed = time.time() - since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Saving model ...')
    model_filename = output_dir + '/best_model.pth'
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)
