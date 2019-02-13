from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict
import numpy as np


__C = edict()
cfg = __C

__C.CONFIG_NAME = 'ConNet'
__C.DATASET_NAME = 'SVHN'
__C.SEED = 1234

# Training options
__C.TRAIN = edict()
__C.TRAIN.DATASET_SPLIT = 'train'
__C.TRAIN.VALID_SPLIT = 0.8
__C.TRAIN.SAMPLE_SIZE = 100
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_EPOCHS = 5
__C.TRAIN.LR = 0.001
__C.TRAIN.MOM = 0.9


def _merge_a_into_b(a, b):
    '''
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.

    Parameters
    ----------
    a : dict
        Config dictionary a.
    b : dict
        Config dictionary b.

    '''
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            # raise KeyError('{} is not a valid config key'.format(k))
            b[k] = v

        else:
            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                      'for config key: {}').format(type(b[k]),
                                                                   type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    _merge_a_into_b(a[k], b[k])
                except Exception as e:
                    print('Error under config key: {}'.format(k))
                    raise e
            else:
                b[k] = v


def cfg_from_file(filename):
    '''
    Load a config file and merge it into the default options.

    Parameters
    ----------
    filename : string
        Path to filename.

    '''
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
