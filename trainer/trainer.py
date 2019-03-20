from __future__ import print_function
import os

from collections import OrderedDict
import copy
from easydict import EasyDict
import operator
from pathlib import Path
import pprint
import ruamel.yaml as yaml
import skopt
import shutil
import time
from tensorboardX import SummaryWriter

import torch

from utils.config import parse_dict, generate_config
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p, save_obj, load_obj
from trainer.utils import (
    load_state_dict, save_state_dict, set_seed,
    define_model, define_optimizer, define_loss,
    define_train_cfg, batch_loop
    )


def train(cfg):
    '''
    Training function.

    Parameters
    ----------
    cfg : dict
        Config dict to use for training.

    Returns
    -------
    valid_best_accuracy : float
        Return the best model accuracy on validation set.

    '''

    # Config
    print('Using config:')
    pprint.pprint(cfg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # Checkpointing
    cfg_filename = os.path.join(cfg['output_dir'], 'cfg.yml')
    state_filename = os.path.join(cfg['output_dir'],
                                  'checkpoint_state.pth.tar')
    state = EasyDict({'model': None,
                      'optim': None,
                      'train': None})

    if Path(state_filename).exists():
        with open(cfg_filename, 'r') as f:
            save_cfg = yaml.load(f)

        if not operator.eq(cfg, save_cfg) and not cfg['skopt']:
            raise AssertionError(
                'The config is not the same as the checkpointing one')

        elif not operator.eq(cfg, save_cfg) and cfg['skopt']:
            save_cfg_iteration = save_cfg['skopt_cfg']['iteration']
            cfg_iteration = cfg['skopt_cfg']['iteration']
            if save_cfg_iteration + 1 == cfg_iteration:
                # Save config
                with open(cfg_filename, 'w') as f:
                    yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)
                print('No checkpoint available...')
            else:
                raise AssertionError(
                    'The config is not the same as the checkpointing one')
        else:
            print('Load checkpoint...')
            state = load_state_dict(state_filename, device)
            cfg['seed'] = state.seed

    else:
        # Save config
        with open(cfg_filename, 'w') as f:
            yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)
        print('No checkpoint available...')

    # Seed
    set_seed(cfg['seed'])

    # Data
    train_loader, valid_loader = prepare_dataloaders(
        cfg['input_dir'],
        cfg['metadata_filename'],
        cfg['dataloader']['batch_size'],
        cfg['dataloader']['valid_split'],
        cfg['dataloader']['sample_size'])

    # Model
    model = define_model(EasyDict(cfg['model']), device, state.model)
    best_model = copy.deepcopy(model)
    print('Using model:')
    print(model)

    # Optimizer
    optimizer = define_optimizer(EasyDict(cfg['optimizer']),
                                 model.parameters(),
                                 state.optim)
    print('Using optimizer:')
    print(optimizer)

    # TensorboardX
    writer = SummaryWriter(log_dir=cfg['output_dir'])

    # Loss
    multiloss = cfg['model']['multiloss']
    loss_function = define_loss(multiloss=multiloss)

    # Training
    train_cfg = define_train_cfg(EasyDict(cfg['train']), state.train)
    print('Start training...')
    # Iterate over epochs
    for epoch in range(train_cfg.starting_epoch, train_cfg.num_epochs):

        print('\n\n\nIterating over training data...')
        model.train()

        train_scores = batch_loop(
            train_loader, model, optimizer,
            loss_function, device,
            multiloss=multiloss, mode='training')

        # Train statistics
        train_loss = train_scores['epoch_loss']
        train_accuracy = train_scores['accuracy']
        if multiloss:
            train_per_branch_accuracy = train_scores['per_branch_accuracy']

        print('Iterating over validation data...')
        model.eval()
        valid_scores = batch_loop(
            valid_loader, model, optimizer,
            loss_function, device,
            multiloss=multiloss, mode='validation')

        # Valid statistics
        valid_loss = valid_scores['epoch_loss']
        valid_accuracy = valid_scores['accuracy']
        if multiloss:
            valid_per_branch_accuracy = valid_scores['per_branch_accuracy']

        # Keep trace of train/valid loss history and valid accuracy
        train_cfg.train_loss_history.append(train_loss)
        train_cfg.train_accuracy_history.append(train_accuracy)
        train_cfg.valid_loss_history.append(valid_loss)
        train_cfg.valid_accuracy_history.append(valid_accuracy)

        print('\nEpoch: {}/{}'.format(epoch + 1, train_cfg.num_epochs))

        # Print train statistics
        print('\tTrain Loss: {:.4f}'.format(train_loss))
        print('\tTrain Accuracy: {:.4f}'.format(train_accuracy))
        if multiloss:
            print('\tTrain Accuracy per branch: {}'.format(
                train_per_branch_accuracy))

        # Print valid statistics
        print('\tValid Loss: {:.4f}'.format(valid_loss))
        print('\tValid Accuracy: {:.4f}'.format(valid_accuracy))
        if multiloss:
            print('\tValid Accuracy per branch: {}'.format(
                valid_per_branch_accuracy))

        writer.add_scalar('data/train_loss', train_loss, epoch)
        writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('data/valid_loss', valid_loss, epoch)
        writer.add_scalar('data/valid_accuracy', valid_loss, epoch)

        # Early stopping and checkpointing best model
        if valid_accuracy > train_cfg.valid_best_accuracy or epoch == 0:
            train_cfg.patience = 0
            train_cfg.valid_best_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)

            print('New best model: checkpointing current state...')
            # This will be redundant in memory with save_state_dict, but useful
            # for loading a model standalone
            torch.save(best_model, cfg['output_dir'] + '/checkpoint.pth')

            train_cfg.epoch_state = epoch + 1

            # save entire state dict, including model, optimizer and config
            save_state_dict(state_filename, device,
                            model, optimizer, train_cfg)

        else:
            # Number of epochs to accept that validation
            # score hasnt increased
            train_cfg.patience += 1

        if train_cfg.patience > train_cfg.max_patience:
            print('Max patience reached, ending training')
            break

    time_elapsed = time.time() - train_cfg.since

    print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best valid accuracy: ', train_cfg.valid_best_accuracy)

    print('Saving best model ...')
    model_filename = os.path.join(cfg['output_dir'], 'best_model.pth')
    torch.save(best_model, model_filename)
    print('Best model saved to :', model_filename)

    valid_best_accuracy = train_cfg.valid_best_accuracy

    return valid_best_accuracy


def train_skopt(cfg, base_estimator='GP',
                n_initial_points=10, random_state=0,
                train_function=train):

    '''
    Do a Bayesian hyperparameter optimization.
    This code was inspired by Francis Dutil work.

    Parameters
    ----------
    cfg : dict
        Configuration dict to use.
    base_estimator : str
        skopt Optimization procedure. In ['GP', 'RF', 'ET', 'GBRT']
        Default 'GP'.
    n_initial_points: int
        Number of random search before starting the optimization.
        Default 10.
    random_state : int
        Seed to use for skopt function.
        Default 0.
    train_function : object
        The trainig procedure to optimize. The function should take
        a dict as input and return a metric to maximize.

    '''

    # Checkpointing
    cfg_filename = os.path.join(cfg['output_dir'], 'cfg_ori.yml')

    if Path(cfg_filename).exists():
        with open(cfg_filename, 'r') as f:
            save_cfg = EasyDict(yaml.load(f))

        if not operator.eq(cfg, save_cfg):
            raise AssertionError(
                'The config is not the same as the checkpointing one')

    else:
        # Save config
        with open(cfg_filename, 'w') as f:
            yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)

    # Sparse the parameters that we want to optimize
    skopt_args = OrderedDict(parse_dict(cfg))
    n_iter = cfg['skopt_cfg']['n_iter']

    # Create the optimizer
    starting_iter = 0
    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator=base_estimator,
                                n_initial_points=n_initial_points,
                                random_state=random_state)

    valid_best_accuracy = 0
    skopt_accuracy_history = []
    skopt_suggestion_history = []

    # Checkpointing
    state_skopt_filename = os.path.join(cfg['output_dir'],
                                        'checkpoint_skopt_state.pkl')

    # Check if checkpointing exist
    if Path(state_skopt_filename).exists():
        print('Load skopt checkpoint...')
        state_skopt = load_obj(state_skopt_filename)
        state_skopt = EasyDict(state_skopt)
        # Reset skopt hyper-parameters
        starting_iter = state_skopt.iteration
        optimizer.rng = state_skopt.optim_state
        valid_best_accuracy = state_skopt.valid_best_accuracy
        skopt_accuracy_history = state_skopt.skopt_accuracy_history
        skopt_suggestion_history = state_skopt.skopt_suggestion_history
    else:
        print('No skopt checkpoint...')

    for iteration in range(starting_iter, n_iter):
        print('\n\n\nStart skopt iteration # {}'.format(iteration))
        suggestion = optimizer.ask()
        skopt_suggestion_history.append(suggestion)
        this_cfg = generate_config(cfg, skopt_args, suggestion)
        this_cfg['skopt_cfg']['iteration'] = iteration
        this_cfg['output_dir'] = os.path.join(
           cfg['output_dir'], 'currrent_iter')
        mkdir_p(this_cfg['output_dir'])

        try:
            # We minimize the negative accuracy/AUC
            valid_accuracy = train_function(this_cfg)
            optimizer.tell(suggestion, - valid_accuracy)
            skopt_accuracy_history.append(valid_accuracy)
        except RuntimeError as e:
            print('''The following error was raised:\n {} \n,
                     launching next experiment.'''.format(e))
            # Something went wrong, (probably a CUDA error).
            valid_accuracy = 0.
            optimizer.tell(suggestion, valid_accuracy)
            skopt_accuracy_history.append(valid_accuracy)

        state_skopt = {'optim_state': optimizer.rng,
                       'iteration': iteration + 1,
                       'valid_best_accuracy': valid_best_accuracy,
                       'skopt_accuracy_history': skopt_accuracy_history,
                       'skopt_suggestion_history': skopt_suggestion_history,
                       'skopt_args': skopt_args}
        save_obj(state_skopt, state_skopt_filename)

        if valid_accuracy > valid_best_accuracy or iteration == 0:
            valid_best_accuracy = valid_accuracy

            print('Checkpointing new best model...')
            path_best_cfg = os.path.join(
                this_cfg['output_dir'], 'cfg.yml')
            path_best_model = os.path.join(
                this_cfg['output_dir'], 'best_model.pth')
            path_checkpoint_state = os.path.join(
                this_cfg['output_dir'], 'checkpoint_state.pth.tar')
            shutil.copy(path_best_cfg, cfg['output_dir'])
            shutil.copy(path_best_model, cfg['output_dir'])
            shutil.copy(path_checkpoint_state, cfg['output_dir'])

    print('\n\n\n### FIND BEST CONFIG/MODEL ###')

    # Find best config/model index
    maxpos = skopt_accuracy_history.index(max(skopt_accuracy_history))

    print('Best skopt config')
    best_cfg_filename = os.path.join(cfg['output_dir'], 'cfg.yml')
    with open(best_cfg_filename, 'r') as f:
        best_cfg = yaml.load(f)
    pprint.pprint(best_cfg)

    print('Best config/model index: {}'.format(maxpos))
    print('Skopt args: {}'.format(list(skopt_args.keys())))
    print('Best skopt suggestion: {}'.format(skopt_suggestion_history[maxpos]))
    print('Best skopt accuracy: {}'.format(skopt_accuracy_history[maxpos]))
