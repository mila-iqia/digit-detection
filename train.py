from __future__ import print_function
import os
import sys

import argparse

from trainer.trainer import train, train_skopt
from utils.config import load_config


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', type=str,
                        default='config/base_config.yml',
                        help='Mendatory config file.')

    parser.add_argument("--metadata_filename", type=str,
                        default='data/SVHN/train_metadata.pkl',
                        help='''metadata_filename will be the absolute
                                 path to the directory to be used for
                                 training.''')

    parser.add_argument('--dataset_dir', type=str,
                        default='data/SVHN/train',
                        help='''Absolute path to the data directory to
                                be used for training.''')

    parser.add_argument('--results_dir', type=str,
                        default='results/',
                        help='''Absolute path to a output directory to
                                be used for training.''')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load the arguments
    args = parse_args()

    # Load the config file
    cfg = load_config(args)

    # Train de model
    if cfg.skopt:
        # Train using hyperparameters tuning with skopt
        train_skopt(cfg)
    else:
        train(cfg)
