import os

import copy
import datetime
import dateutil.tz
import ruamel.yaml as yaml
from skopt.space import Real, Integer, Categorical

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
        cfg = yaml.load(f)
    return cfg


def load_config(args):
    '''
    Load the config from args.

    Parameters
    ----------
    args : dict
        {'cfg': 'file.yml',
         'dataset_dir': str, path to data,
         'metadata_filename': str, path to meta data,
         'results_dir': str, path to results}

    Returns
    -------
    cfg : dict
        Config dict.

    '''

    if args.cfg is None:
        raise Exception('No config file specified.')

    cfg = cfg_from_file(args.cfg)

    # Checkpointing
    if 'output_dir' not in cfg:
        # Add metadata_filename to cfg
        cfg['metadata_filename'] = args.metadata_filename

        # Add input_dir to cfg
        cfg['input_dir'] = args.dataset_dir

        # Add output_dir to cfg
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        cfg['output_dir'] = os.path.join(
            args.results_dir,
            '%s_%s_%s' % (cfg['dataset_name'], cfg['config_name'], timestamp))
        mkdir_p(cfg['output_dir'])

        # Overwrite args.cfg file for checkpointing purpose
        with open(cfg['output_dir']+'/cfg.yml', 'w') as f:
            yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)

    return cfg


def parse_dict(d_, prefix='', lst=[]):
    '''
    Function to sparse the yaml config file. Find the keys in the
    config dict that are to be optimized.

    Parameters
    ----------
    d_ : dict
        Object which can be edict.
    prefix : str
        Prefix.
        Default: start with ''.
    lst : list
        List of keys to be optimized.
        Default: start with[].

    Returns
    -------
    lst : list
        List of keys to be optimized.

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

        except Exception:
            pass

        return lst


def set_key(dic, key, value):
    '''
    Aux function to set the value of a key in a dict.

    Parameters
    ----------
    dic : dict
        Config dictionary.
    key : str
        Config key.
    value : float
        Config value.

    '''
    k1 = key.split(".")
    k1 = list(filter(lambda l: len(l) > 0, k1))
    if len(k1) == 1:
        dic[k1[0]] = value.item()
    else:
        set_key(dic[k1[0]], ".".join(k1[1:]), value)


def generate_config(config, keys, new_values):
    '''
    Generate a new config from the config containing hyper-parameters to
    optimize.

    Parameters
    ----------
    config : dict
        Config dictionary.
    keys : str
        Config keys.
    new_values :
        Config new values.

    Returns
    -------
    new_config : dict
        New config dictionary.

    '''
    new_config = copy.deepcopy(config)
    for i, key in enumerate(list(keys.keys())):
        set_key(new_config, key, new_values[i])
    return new_config
