"""
Make Model script

Makes a BayesFusion SMILE model that can later easily be loaded in.
"""

# region Preliminaries
import json
import os
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import warnings

from os.path import dirname
from sklearn.exceptions import UndefinedMetricWarning

# Custom imports
root_dir = dirname(dirname(__file__))
for e in {'libs', 'src'}:
    sys.path.append(os.path.join(root_dir, e))

import PxS as pxs
import PxL as pxl
import mercs as mercs

from exp.utils.filesystem import ensure_dir
from exp.utils.extra import debug_print  # For debugging purposes
# endregion

VERBOSITY = 1
PRECISION = 6
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def main(config_fname, fold):

    # Load config
    config = load_config(config_fname)

    # Configuration matters
    io_config = config['io']
    mod_config = config.get("mod", {})

    fit_config = config.get("fit", {})
    machine = config.get("machine", None)

    io_config['fold'] = fold

    msg = """
    io_config: {}
    """.format(io_config)
    debug_print(msg, V=VERBOSITY, l=2)

    # Loading things
    train_fname = load_input(io_config)

    # Induce
    model = induction(train_fname,
                      fit_config,
                      mod_config,
                      io_config,
                      machine=machine)

    # Save Model + Config
    cfg = {'mod':   mod_config,
           'fit':   fit_config}
    save_model_and_model_config(model, io_config, cfg)

    return 0


# Inputs
def load_config(config_fname):
    with open(config_fname, 'r') as f:
        config = json.load(f)
    return config


def load_input(io_config):
    """
    Selects the correct inputfiles based on fold.

    Parameters
    ----------
    io_config

    Returns
    -------

    """

    fold = io_config['fold']

    train_data_fnames = io_config['file']['train_data']

    train_fname = [t[1] for t in train_data_fnames
                   if t[0] == fold][0]

    return train_fname


# Actions
def induction(train_fname, fit_config, mod_config, io_config, machine=None):
    # Config
    mod_type = mod_config['type']
    fold = io_config['fold']
    cwd = io_config['dirs']['prod-tmp']

    # Actions
    if mod_type in {'PxS'}:
        net_fname = [t[1] for t in io_config['file']['net']
                     if t[0] == fold][0]
        ensure_dir(dirname(net_fname), empty=False)

        model = pxs.PxS()

        msg = """
        Succesfully initialized PxS model.
        
        Net fname: {}
        """.format(net_fname)
        debug_print(msg, V=VERBOSITY)

        res = model.fit(train_fname,
                        model_fname=net_fname,
                        cwd=cwd,
                        **fit_config)

        if res != 0:
            msg = """
            Code returned from fit method: {}
            """.format(res)
            raise ValueError(msg)

    elif mod_type in {'PxL'}:
        net_fname = [t[1] for t in io_config['file']['net']
                     if t[0] == fold][0]
        ensure_dir(dirname(net_fname), empty=False)

        model = pxl.PxL(cwd=cwd, machine=machine)

        msg = """
        Succesfully initialized PxL model.

        Net fname: {}
        """.format(net_fname)
        debug_print(msg, V=VERBOSITY)

        res = model.fit(i=train_fname, o=net_fname, **fit_config)

        if res != 0:
            msg = """
            Code returned from fit method: {}
            """.format(res)
            raise ValueError(msg)

    elif mod_type in {'Mercs'}:
        train = pd.read_csv(train_fname)
        model = mercs.MERCS()
        model.fit(train, **fit_config, delimiter='.')

    else:
        msg = """
            Did not recognize model type: {}
            """.format(mod_type)
        raise ValueError(msg)

    return model


# Outputs
def save_model_and_model_config(model, io_config, mod_config):
    # Config
    fold = io_config['fold']
    mod_config_fname = io_config['file']['mod-config']
    mod_fname = [t[1] for t in io_config['file']['mod']
                 if t[0] == fold][0]

    # Actions
    ensure_dir(dirname(mod_config_fname), empty=False)
    with open(mod_config_fname, 'w') as f:
        json.dump(mod_config, f, indent=4)

    ensure_dir(dirname(mod_fname), empty=False)
    with open(mod_fname, 'wb') as f:
        pkl.dump(model, f)

    msg = """
    Successful save of model to: {}
    """.format(mod_fname)
    debug_print(msg, V=VERBOSITY)

    return


# For executable script
if __name__ == '__main__':
    config_fname_outer_scope = sys.argv[1]
    fold_outer_scope = int(sys.argv[2])

    main(config_fname_outer_scope, fold_outer_scope)
