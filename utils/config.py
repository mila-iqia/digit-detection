import os

import copy
from easydict import EasyDict
from skopt.space import Real, Integer, Categorical
import yaml

from utils.misc import mkdir_p


def cfg_from_file(filename):
    '''
    Load a .yml config file.

    Parameters
    ----------
    filename : string
        Path to filename.

    Returns
    -------
    cfg : dict
        A dict with the config.

    '''
    with open(filename, 'r') as f:
        cfg = EasyDict(yaml.load(f))
    return cfg


def load_config(args):
    '''
    Load the config from args.

    Parameters
    ----------
    args : dict
        {'cfg': 'file.yml',
         'dataset_dir': 'path_to_data',
         'results_dir': path to results}

    Returns
    -------
    cfg : dict
        Config dict.

    '''

    if args.cfg is None:
        raise Exception('No config file specified.')

    cfg = cfg_from_file(args.cfg)

    cfg.metadata_filename = args.metadata_filename

    cfg.input_dir = args.dataset_dir
    cfg.output_dir = os.path.join(
        args.results_dir,
        '%s_%s' % (cfg.dataset_name, cfg.config_name))

    mkdir_p(cfg.output_dir)

    return cfg


def parse_dict(d_, prefix='', lst=[]):
    '''
    Helper function to help us sparse the yaml config file.
    Find the keys in the config dict that are to be optimized.
    '''
    if isinstance(d_, dict):
        for key in d_.keys():
            temp = parse_dict(d_[key], prefix + '.' + key, [])
            if temp:
                lst += temp
        return lst
    else:
        try:
            x = eval(d_)
            if isinstance(x, (Real, Integer, Categorical)):
                lst.append((prefix, x))
        except:
            pass
        return lst


def set_key(dic, key, value):
    '''
    Aux function to set the value of a key in a dict
    '''
    k1 = key.split(".")
    k1 = list(filter(lambda l: len(l) > 0, k1))
    if len(k1) == 1:
        dic[k1[0]] = value
    else:
        set_key(dic[k1[0]], ".".join(k1[1:]), value)


def generate_config(config, keys, new_values):
    new_config = copy.deepcopy(config)
    for i, key in enumerate(list(keys.keys())):
        set_key(new_config, key, new_values[i])
    return new_config
