#!/usr/bin/env python

from __future__ import print_function
import os
import sys

import argparse
import dateutil.tz
import datetime
import numpy
import pprint
import random
from shutil import copyfile

import torch

from utils.config import (cfg, cfg_from_file)
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p
from models.models import BaselineCNN
from trainer.trainer import train_model


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', dest='cfg_file', type=str,
                        default=None,
                        help='optional config file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # Load the config file
    args = parse_args()

    if args.cfg_file:
        args.cfg_file = os.path.join('config', args.cfg_file)
        cfg_from_file(args.cfg_file)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    print('timestamp: {}'.format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.INPUT_DIR = os.path.join(
        cfg.DATA_DIR, cfg.DATASET_NAME, cfg.TRAIN.DATASET_SPLIT)
    cfg.METADATA_FILENAME = os.path.join(
        cfg.DATA_DIR, cfg.DATASET_NAME, cfg.METADATA_FILENAME)
    cfg.OUTPUT_DIR = os.path.join(
        cfg.RESULTS_DIR,
        '%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp))

    mkdir_p(cfg.OUTPUT_DIR)

    if args.cfg_file:
        copyfile(args.cfg_file, os.path.join(cfg.OUTPUT_DIR, 'comfig.yml'))

    print('Data dir: {}'.format(cfg.INPUT_DIR))
    print('Output dir: {}'.format(cfg.OUTPUT_DIR))

    print('Using config:')
    pprint.pprint(cfg)

    # make the results reproductible
    print('pytorch/random seed: {}'.format(cfg.SEED))
    random.seed(cfg.SEED)
    numpy.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # Prepare data
    (train_loader,
     valid_loader) = prepare_dataloaders(dataset_split=cfg.TRAIN.DATASET_SPLIT,
                                         dataset_path=cfg.INPUT_DIR,
                                         metadata_filename=cfg.METADATA_FILENAME,
                                         batch_size=cfg.TRAIN.BATCH_SIZE,
                                         sample_size=cfg.TRAIN.SAMPLE_SIZE,
                                         valid_split=cfg.TRAIN.VALID_SPLIT)

    # Define model architecture
    baseline_cnn = BaselineCNN()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    train_model(baseline_cnn,
                train_loader=train_loader,
                valid_loader=valid_loader,
                num_epochs=cfg.TRAIN.NUM_EPOCHS,
                device=device,
                model_filename=cfg.OUTPUT_DIR)
