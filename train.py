import os
import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/SVHN')
    parser.add_argument("--results_dir", type=str, default='results')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default='train')

    args = parser.parse_args()
    datadir = args.data_dir
    results_dir = args.results_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    sample_size = args.sample_size
    model_filename = args.model_filename
    dataset_split = args.dataset_split

    #  sample_size = None
    if model_filename:
        model_filename = results_dir + '/' + model_filename

    (train_loader,
     valid_loader) = prepare_dataloaders(dataset_split=dataset_split,
                                         batch_size=batch_size,
                                         sample_size=sample_size,
                                         datadir=datadir)

    # Define model architecture
    baseline_cnn = BaselineCNN()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    train_model(baseline_cnn,
                train_loader=train_loader,
                valid_loader=valid_loader,
                num_epochs=num_epochs,
                device=device,
                model_filename=model_filename)
