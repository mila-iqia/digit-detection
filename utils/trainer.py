from collections import defaultdict
from easydict import EasyDict
import numpy as np
import random
import time

import torch
from tqdm import tqdm

from models.baselines import BaselineCNN, BaselineCNN_dropout
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vgg import VGG
from models.multiloss import FC_Layer, MultiLoss


def load_state_dict(filename, device):
    if device == 'cuda:0':
        state = torch.load(filename)
    else:
        state = torch.load(filename,
                           map_location=lambda storage,
                           loc: storage)
    if state.device != device:
        raise Exception('Experiment started with a different device.')
    return state


def save_state_dict(filename, device, model, optimizer, train_cfg):
    state = EasyDict(defaultdict(dict))
    state.seed = {'np_state': np.random.get_state(),
                  'random_state': random.getstate(),
                  'torch_state': torch.get_rng_state()
                  }
    if torch.cuda.is_available():
        state.seed.torch_state_cuda = torch.cuda.get_rng_state()
    state.device = device
    state.model = model.state_dict()
    state.optim = optimizer.state_dict()
    state.train = train_cfg
    torch.save(state, filename)


def set_seed(seed):
    if isinstance(seed, int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    elif isinstance(seed, dict):
        np.random.set_state(seed.np_state)
        random.setstate(seed.random_state)
        torch.set_rng_state(seed.torch_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(seed.torch_state_cuda)
    else:
        raise Exception('Seed must be a int or a dict.')


def define_model(model_cfg, device, model_state=None):
    num_classes = model_cfg.num_classes
    if model_cfg.model == 'BaselineCNN':
        model = BaselineCNN(num_classes)
    elif model_cfg.model == 'BaselineCNN_dropout':
        model = BaselineCNN_dropout()
    elif model_cfg.model == 'ResNet18':
        model = ResNet18(num_classes)
    elif model_cfg.model == ' ResNet34':
        model = ResNet34(num_classes)
    elif model_cfg.model == ' ResNet50':
        model = ResNet50(num_classes)
    elif model_cfg.model == ' ResNet101':
        model = ResNet101(num_classes)
    elif model_cfg.model == ' ResNet152':
        model = ResNet152(num_classes)
    elif model_cfg.model == 'VGG':
        model = VGG(num_classes)
    elif model_cfg.model == 'MultiLoss':
        base_net = VGG('VGG11', classify=False)
        model = MultiLoss(base_net, FC_Layer)
        # TODO add cfg.multiloss variable
    else:
        raise Exception('The model specified is not avaiable.')

    model = model.to(device)
    if model_state:
        model.load_state_dict(model_state)
    return model


def define_optimizer(optim_cfg, model_param, device, optim_state=None):
    optimizer = torch.optim.SGD(
        model_param, lr=optim_cfg.lr)  # , momentum=optim_cfg.momentum)
    if optim_state:
        optimizer.load_state_dict(optim_state)
    return optimizer


def define_loss(multiloss):

    # TODO replace multiloss with cfg.multiloss
    if multiloss:
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    return loss_function


def define_train_cfg(train_cfg, train_state=None):
    if not train_state:
        train_cfg.starting_epoch = 0
        train_cfg.patience = 0
        train_cfg.max_patience = 8
        train_cfg.train_loss_history = []
        train_cfg.train_accuracy_history = []
        train_cfg.valid_loss_history = []
        train_cfg.valid_accuracy_history = []
        train_cfg.valid_best_accuracy = 0
        train_cfg.since = time.time()
    else:
        train_cfg = train_state
    return train_cfg


def array_to_housenumber(housenum_array):

    '''
    Convert an array (like predictions and targets) to their number
    equivalent.

    returns an ndarray.
    '''
    house_numbers = []

    for idx, seq_len in enumerate(housenum_array[:, 0]):

        # Get only predicted sequence as single number
        house_number_arr = housenum_array[idx, 1:seq_len+1]
        if seq_len == 0:
            # Network predicts no house number
            house_number = -1
        else:
            house_number = int("".join(house_number_arr.astype(str)))
        house_numbers.append(house_number)

    # Convert to ndarray
    house_numbers = np.asarray(house_numbers)
    return house_numbers


def batch_loop(loader, model, optimizer, loss_function, device, train=True,
               multiloss=True):

    tot_loss = 0
    n_iter = 0
    correct = 0
    n_samples = 0

    for i, batch in enumerate(tqdm(loader)):
        # get the inputs
        loss = 0
        inputs, targets = batch['image'], batch['target']
        inputs = inputs.to(device)
        target_ndigits = targets[:, 0].long()
        target_ndigits = target_ndigits.to(device)

        if train:
            # Zero the gradient buffer
            optimizer.zero_grad()

        # Forward
        outputs = model(inputs)

        # Iterate through each target and compute the loss
        batch_preds = []
        if multiloss:

            for index in range(targets.shape[1]):
                target = targets[:, index].long()
                target = target.to(device)
                pred = outputs[index]
                loss += loss_function(pred, target)

                _, predicted = torch.max(pred.data, 1)
                batch_preds.append(predicted)
            batch_preds = torch.stack(batch_preds)  # Combine all results to one tensor
            batch_preds = batch_preds.transpose(1, 0)  # Get same shape as target
            batch_preds = batch_preds.cpu().numpy().astype('int')
            batch_targets = targets.cpu().numpy().astype('int')

        else:
            loss = loss_function(outputs, target_ndigits)

        if train:
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()

        # Statistics
        tot_loss += loss.item()
        n_iter += 1

        n_samples += targets.size(0)

        if multiloss:
            predicted_house_numbers = array_to_housenumber(batch_preds)
            target_house_numbers = array_to_housenumber(batch_targets)

            correct += np.sum(
                target_house_numbers == predicted_house_numbers)

        else:
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target_ndigits).sum().item()

    epoch_loss = tot_loss / n_iter
    accuracy = correct / n_samples
    return epoch_loss, accuracy


