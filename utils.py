import numpy as np
import cPickle as pickle
import os
import json

def save_model(f, model):
    output_folder = os.path.dirname(f)
    try:
        os.makedirs(output_folder)
    except Exception:
        pass
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, 'wb'))


def load_model(f, model):
    ps = pickle.load(open(f, 'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model

## load json
def load_config(cfg_filename):
    '''Load a configuration file.'''
    with open(cfg_filename) as f:
        args = json.load(f)
    return args


def merge_dict(cfg_defaults, cfg_user):
    for k, v in cfg_defaults.items():
        if k not in cfg_user:
            cfg_user[k] = v
        elif isinstance(v, dict):
            merge_dict(v, cfg_user[k])


def load_config_with_defaults(cfg_filename, cfg_default_filename):
    """Load a configuration with defaults."""
    cfg_defaults = load_config(cfg_default_filename)
    cfg = load_config(cfg_filename)
    if cfg_filename != cfg_default_filename:
        merge_dict(cfg_defaults, cfg)
    return cfg
# end load json

