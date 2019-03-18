from collections import defaultdict
from easydict import EasyDict
import numpy as np
import random
import time

import torch
from tqdm import tqdm

from models.resnet import ResNet
from models.vgg import VGG
from models.multiloss import FC_Layer, MultiLoss


def load_state_dict(filename, device):
    '''
    Load the dictionary containing the state of training.

    Parameters
    ----------
    filename : str
        Path to the file where the state dictionary is save.

    device : str
        The device that is used. In ['cuda:0', 'cpu'].

    Returns
    -------
    state : dict
        Dictionary containing the state of training.

    '''
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
    '''
    Save the dictionary containing the state of training.

    Parameters
    ----------
    filename : str
        Path to the file where the state dictionary will be save.
    device : str
        The device that is used. In ['cuda:0', 'cpu']
    model : obj
        The model that we want to save the state.
    optimizer : obj
        The optimizer that we want to save the state.
    train_cfg: dict
        The dictionary containing training configuration and state.

    '''
    state = EasyDict(defaultdict(dict))
    state.seed = {'np_state': np.random.get_state(),
                  'random_state': random.getstate(),
                  'torch_state': torch.get_rng_state()
                  }
    if torch.cuda.is_available():
        state.seed.torch_state_cuda = torch.cuda.get_rng_state()
    state.device = device
    state.model = model.state_dict()
    # Fix a bug when saving optimizer.state_dict()
    if optimizer.state_dict()['state'].keys():
        dict_ = {}
        for k, v in optimizer.state_dict().items():
            if type(v) is dict and v.keys():
                dict_[k] = {str(k2): v2 for k2, v2 in v.items()}
            else:
                dict_[k] = v
        state.optim = dict_
    else:
        state.optim = optimizer.state_dict()
    state.optim = dict_

    state.train = train_cfg
    torch.save(state, filename)


def set_seed(seed):
    '''
    Set the seed.

    Parameters
    ----------
    seed : int or dict.
        If int the seed to be used.
        If dict contain all seed state that need to be fixed.

    '''
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
    '''
    Define the model to be used.

    Parameters
    ----------
    model_cfg : dict
        Dictionary containing the config of the model.
    device : str
        The device that is used. In ['cuda:0', 'cpu']
    model_state : dict
        The state of the model to be load. Default: None.

    Returns
    -------
    model : obj
        The model object.

    '''
    model = model_cfg.model
    multiloss = model_cfg.multiloss
    num_classes = model_cfg.num_classes

    classify = True
    if multiloss:
        classify = False

    if model.startswith('ResNet'):
        base_net = ResNet(model, num_classes, classify)
    elif model.startswith('VGG'):
        base_net = VGG(model, num_classes, classify)
    else:
        raise Exception('The model specified is not avaiable.')

    if multiloss:
        in_dim_fclayer = 512
        if model in ['ResNet50', 'ResNet101', 'ResNet152']:
            in_dim_fclayer = 2048
        model = MultiLoss(base_net, FC_Layer, in_dim_fclayer)
    else:
        model = base_net

    model = model.to(device)
    if model_state:
        model.load_state_dict(model_state)
    return model


def define_optimizer(optim_cfg, model_param, optim_state=None):
    '''
    Define the optimizer to be used.

    Parameters
    ----------
    optim_cfg : dict
        Dictionary containing the config of the optimizer.
    model_param : dict
        The parameters of the model to optimize.
    device : str
        The device that is used. In ['cuda:0', 'cpu']
    optim_state : dict
        The state of the optimizer to be load. Default: None.

    Returns
    -------
    optimizer : obj
        The optimizer object.

    '''
    optimizer = torch.optim.SGD(
        model_param, lr=optim_cfg.lr, momentum=optim_cfg.momentum)
    if optim_state:
        # Fix a bug when loading optimizer.state_dict()
        if optim_state['state'].keys():
            dict_ = {}
            for k, v in optim_state.items():
                if type(v) is EasyDict and v.keys():
                    dict_[k] = {int(k2): v2 for k2, v2 in v.items()}
                else:
                    dict_[k] = v
            optim_state = dict_
        optimizer.load_state_dict(optim_state)
    return optimizer


def define_loss(multiloss):
    '''
    Define the loss.

    Parameters
    ----------
    multiloss : bool
        if True use the multiloss.

    Returns
    -------
    loss_function : function
        The loss function.

    '''
    if multiloss:
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    return loss_function


def define_train_cfg(train_cfg, train_state=None):
    '''
    Define the training config.

    Parameters
    ----------
    train_cfg : dict
        The dictionaty containing the training config.
    train_state : dict
        Dictionary containing the config and the state of the training loop.
        Default None.

    Returns
    -------
    train_cfg : dict
        Dictionary containing the config and the state of the training loop.

    '''
    if not train_state:
        train_cfg.starting_epoch = 0
        train_cfg.patience = 0
        train_cfg.train_loss_history = []
        train_cfg.train_accuracy_history = []
        train_cfg.valid_loss_history = []
        train_cfg.valid_accuracy_history = []
        train_cfg.valid_best_accuracy = 0
        train_cfg.since = time.time()
    else:
        train_cfg = train_state
        train_cfg.starting_epoch = train_cfg.epoch_state
    return train_cfg


def array_to_housenumber(housenum_array):

    '''
    Convert an ndarray (like predictions and targets) to their number
    equivalent.

    Parameters
    ----------
    housenum_array : ndarray
        ndarray containing the predictions and targets for the
        multi-loss model.

    Returns
    -------
    house_numbers : ndarray
        ndarray containing the house number.

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


def batch_loop(loader, model, optimizer, loss_function, device,
               multiloss=True, mode='training'):
    '''
    Define the mini-batch loop.

    Parameters
    ----------
    loader : obj
        The dataloader to iterate.
    model : obj
        The model to use.
    optimizer : obj
        The optimizer to use.
    loss_function : function
        The loss function to use.
    device : str
        The device that is used. In ['cuda:0', 'cpu']
    multiloss = bool
        If true compute the multi-loss.
    mode : str
        The mode to use. In ['training', 'validation', 'testing']

    Returns
    -------
    if mode in ['training', 'validation']:
        epoch_loss : float
            The loss.
        accuracy : float.
            The accuracy.
    elif mode == 'testing':
        accuracy : float.
            The accuracy.
        total_predicted_house_numbers : list
            List containing the predicted house number.

    '''

    assert mode in ['training', 'validation', 'testing'], \
        "mode can only be 'training' or 'testing'"

    tot_loss = 0
    n_iter = 0
    correct = 0
    n_samples = 0
    per_branch_correct = np.zeros((6))
    total_predicted_house_numbers = []

    for i, batch in enumerate(tqdm(loader)):
        # Get the inputs
        loss = 0
        inputs, targets = batch['image'], batch['target']
        inputs = inputs.to(device)
        target_ndigits = targets[:, 0].long()
        target_ndigits = target_ndigits.to(device)

        if mode == 'training':
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

                if mode in ['training', 'validation']:
                    loss += loss_function(pred, target)

                _, predicted = torch.max(pred.data, 1)
                batch_preds.append(predicted)
            # Combine all results to one tensor
            batch_preds = torch.stack(batch_preds)
            # Get same shape as target
            batch_preds = batch_preds.transpose(1, 0)
            batch_preds = batch_preds.cpu().numpy().astype('int')
            batch_targets = targets.cpu().numpy().astype('int')

        else:
            if mode in ['training', 'validation']:
                loss = loss_function(outputs, target_ndigits)

        if mode == 'training':
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()

        # Statistics
        if mode in ['training', 'validation']:
            tot_loss += loss.item()

        n_iter += 1

        n_samples += targets.size(0)

        if multiloss:
            predicted_house_numbers = array_to_housenumber(batch_preds)
            target_house_numbers = array_to_housenumber(batch_targets)

            correct += np.sum(
                target_house_numbers == predicted_house_numbers)

            per_branch_correct += np.sum(batch_preds == batch_targets, axis=0)

            total_predicted_house_numbers.extend(predicted_house_numbers)

        else:
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target_ndigits).sum().item()

    epoch_loss = tot_loss / n_iter
    accuracy = correct / n_samples

    if multiloss:
        per_branch_accuracy = per_branch_correct / n_samples

    if mode in ['training', 'validation']:
        scores = {}
        scores['epoch_loss'] = epoch_loss
        scores['accuracy'] = accuracy
        if multiloss:
            scores['per_branch_accuracy'] = per_branch_accuracy
        return scores

    elif mode == 'testing':
        scores = {}
        scores['accuracy'] = accuracy
        scores['total_predicted_house_numbers'] = total_predicted_house_numbers
        if multiloss:
            scores['per_branch_accuracy'] = per_branch_accuracy
        return scores
