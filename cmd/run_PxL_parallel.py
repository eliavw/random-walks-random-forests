"""
Run PxL in parallel
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning

# Custom imports
root_dir = dirname(dirname(__file__))
for e in {'libs', 'src'}:
    sys.path.append(os.path.join(root_dir, e))

import PxL as pxl

from exp.query.encoding import codes_to_query
from exp.utils.extra import debug_print  # For debugging purposes
# endregion

VERBOSITY = 0
PRECISION = 6
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def main(config_fname, fold, q_idx):
    """
    Run a certain system.

    Stages
    ^^^^^^

        1. Load configurations
            In general, we hold on to a `program(config)` kind of philosophy,
            meaning that in principle, we expect this script to be called with
            a single argument which is a configfile. In this configuration file,
            all the details of how the script is supposed to function are to be
            included.

            Hence, the first step is to extract these configurations. Different
            configurations come from different places, but all of them are dicts,
            saved as json files.
        1.b Alter configurations
            A few important parameters are passed on as cmd-parameters, such as
            fold and query-idx. This because it would be overkill to generate
            separate configfiles for each of those. For consistency later on,
            we explicitly put them in the configurations as well.
        2. Induction
            We induce or load our predictive model. Before doing so, we already
            extract some relevant parameters from the config files.
        3. Inference
            We use our model to perform inference tasks (predictions.)


    Parameters
    ----------
    config_fname
    fold
    q_idx

    Returns
    -------

    """

    # Load config (all different categories)
    config = load_config(config_fname)

    io_config = config['io']
    pxl_config = config.get('PxL', {})

    fit_config = {k.replace('fit.', ''):v for k,v in pxl_config.items()
                  if "fit." in k}
    pred_config = {k.replace('predict.', ''): v for k, v in pxl_config.items()
                  if "predict." in k}

    # Alter configuration with cmd-parameters
    io_config['fold'] = fold
    io_config['load'] = io_config['file'].get('load-mod', None) is not None # If this is non-empty, this is set to true.
    io_config['q_idx'] = q_idx

    msg = """
    io_config: {}
    """.format(io_config)
    debug_print(msg, V=VERBOSITY, level=2)

    # Extract relevant parameters
    train_fname, test_fname = load_input(io_config)
    cwd = io_config['dirs']['prod-tmp']
    machine = config.get("machine", None)
    qry_codes = load_qry_codes(io_config)

    # Induce
    if io_config['load']:
        msg = """
        Loading model from disk; {}
        """.format(io_config['file']['load-mod'])
        debug_print(msg, V=VERBOSITY)
        model, ind_time = load_mod(io_config, cwd, machine)
    else:
        model, ind_time = induction(train_fname, cwd, machine, fit_config)

    # Inference + Evaluation
    eval_config = config['eval']
    inference_and_evaluation(model,
                             test_fname,
                             qry_codes,
                             pred_config,
                             eval_config,
                             io_config,
                             ind_time)

    return


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
    test_data_fnames = io_config['file']['test_data']

    train_fname = [t[1] for t in train_data_fnames
                   if t[0] == fold][0]
    test_fname = [t[1] for t in test_data_fnames
                  if t[0] == fold][0]

    return train_fname, test_fname


def load_qry_codes(io_config):
    qry_codes_fname = io_config['file']['qry-codes']
    qry_codes = np.load(qry_codes_fname)

    return qry_codes


def load_mod(io_config, cwd, machine):
    fold = io_config['fold']

    mod_fnames = io_config['file']['load-mod']
    mod_fname = [t[1] for t in mod_fnames
                 if t[0] == fold][0]

    msg = """
    Loading an external model:      {}
    """.format(mod_fname)
    debug_print(msg, V=VERBOSITY)

    with open(mod_fname, 'rb') as f:
        mod = pkl.load(f)

    mod.set_cwd(cwd)
    mod.set_machine(machine=machine)

    ind_time = mod.s['model_data']['ind_time']
    ind_time = round(ind_time, PRECISION)

    return mod, ind_time


# Actions
def induction(train_fname, cwd, machine, cfg_fit):
    model = pxl.PxL(cwd=cwd, machine=machine)

    msg = """
    Succesfully initialized PxL model.

    Cfg_fit:            {}
    """.format(cfg_fit)
    debug_print(msg, V=VERBOSITY)

    res = model.fit(i=train_fname, **cfg_fit)

    msg = """
    Code returned from fit method: {}
    """.format(res)
    debug_print(msg, V=VERBOSITY)

    ind_time = model.s['model_data']['ind_time']
    ind_time = round(ind_time, PRECISION)

    return model, ind_time


def inference_and_evaluation(model,
                             test_fname,
                             qry_codes,
                             pred_config,
                             eval_config,
                             io_config,
                             ind_time):
    # Extract config
    q_idx = io_config['q_idx']
    _, q_targ, q_miss = codes_to_query(qry_codes)
    eval_kinds = eval_config.get('kinds', ['macro_f1'])

    msg = """
    {}
    eval_kinds: {}
    """.format(__file__, qry_codes, eval_kinds)
    debug_print(msg, V=VERBOSITY)

    # Initialize
    head_tuple = ('q_idx', *eval_kinds)
    data_tuple = tuple([0] + [0.0] * len(eval_kinds))
    tuple_list = [data_tuple]

    # Actions
    pred_config['miss_idx'] = np.array(q_miss[q_idx]).tolist()  # For converting types.
    pred_config['targ_idx'] = np.array(q_targ[q_idx]).tolist()
    pred_config['q_idx'] = q_idx

    msg = """
    cfg_pred:   {}
    """.format(pred_config)
    debug_print(msg, V=VERBOSITY)

    test_data = pd.read_csv(test_fname, header=None)
    true_data = test_data.values[:, q_targ[q_idx]]

    pred_data = model.predict(test_fname,
                              **pred_config)

    if not isinstance(pred_data, np.ndarray):
        # If all goes wrong, fill in np.NaN
        pred_data = np.full_like(true_data, fill_value=np.nan)

    # region Advanced diagnostics for debugging.
    if VERBOSITY > 0:
        uvals_pred_data = np.unique(pred_data)
        msg = """
        Diagnostics about pred_data.
        Unique values:  {}
        Shape:          {}
        Type:           {}

        Diagnostics about true_data.
        Shape:          {}
        Type:           {}
        """.format(uvals_pred_data, pred_data.shape, pred_data.dtype, true_data.shape, true_data.dtype)
        debug_print(msg, V=VERBOSITY)
    # endregion

    # Evaluation
    evals = [round(eval_dict[kind](true_data, pred_data), PRECISION)
             for kind in eval_kinds]

    # Collect results
    tuple_list[0] = q_idx, *evals
    results_df = pd.DataFrame.from_records(tuple_list, columns=head_tuple)

    # Collect timings
    inf_timing = round(model.s['model_data']['inf_time'], PRECISION)
    timings_tuple = (q_idx, ind_time, inf_timing)
    timings_df = pd.DataFrame.from_records([timings_tuple],
                                           columns=('q_idx', 'ind_time', 'inf_time'))

    # Write results
    save_results(results_df, io_config)
    save_timings(timings_df, io_config)

    return


# Outputs
def save_results(results_df, io_config):
    fold = io_config['fold']
    q_idx = io_config['q_idx']
    results_fnames = io_config['file']['results']

    results_fname = [t[2] for t in results_fnames
                     if t[0] == fold
                     if t[1] == q_idx][0]

    results_df.to_csv(results_fname, index=False)

    return


def save_timings(timings_df, io_config):
    fold = io_config['fold']
    q_idx = io_config['q_idx']
    timings_fnames = io_config['file']['timings']

    timings_fname = [t[2] for t in timings_fnames
                     if t[0] == fold
                     if t[1] == q_idx][0]

    timings_df.to_csv(timings_fname, index=False)

    return


# Evaluation
def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def weigh_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# Dict instead of entire class, serves our purpose => OK.
eval_dict = {'macro_f1': macro_f1,
             'micro_f1': micro_f1,
             'weigh_f1': weigh_f1,
             'accuracy': accuracy}

# For executable script
if __name__ == '__main__':
    config_fname_outer_scope = sys.argv[1]
    fold_outer_scope = int(sys.argv[2])
    q_idx_outer_scope = int(sys.argv[3])

    main(config_fname_outer_scope, fold_outer_scope, q_idx_outer_scope)
