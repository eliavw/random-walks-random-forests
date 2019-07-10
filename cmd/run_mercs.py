"""
Run mercs script.

Trains, tests and evaluates a mercs model.
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

import mercs

from exp.query.encoding import codes_to_query
from exp.utils.extra import debug_print  # For debugging purposes
# endregion

VERBOSITY = 1
PRECISION = 6
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def main(config_fname, fold):

    # Load config
    config = load_config(config_fname)

    # Load external files
    io_config = config['io']
    mercs_config = config['Mercs']

    io_config['fold'] = fold

    train_data, test_data = load_input(io_config)
    qry_codes = load_qry_codes(io_config)

    # Induce
    if io_config['file'].get('load-mod', False):
        model, ind_time = load_mod(io_config)
    else:
        model, ind_time = induction(train_data, mercs_config)

    # Inference + Evaluation
    eval_config = config['eval']
    results_df, inf_timing = inference_and_evaluation(model,
                                                      test_data,
                                                      qry_codes,
                                                      mercs_config,
                                                      eval_config)

    # Tidy timings
    timings_df = tidy_timings(ind_time, inf_timing)

    # Save output
    save_results(results_df, io_config)
    save_timings(timings_df, io_config)
    return 0


# Inputs
def load_config(config_fname):
    with open(config_fname, 'r') as f:
        config = json.load(f)
    return config


def save_config(config, config_fname):
    with open(config_fname, 'w') as f:
        json.dump(config, f)
    return


def load_input(io_config):
    fold = io_config['fold']

    train_data_fnames = io_config['file']['train_data']
    test_data_fnames = io_config['file']['test_data']

    train_data_fname = [t[1] for t in train_data_fnames
                        if t[0] == fold][0]
    test_data_fname = [t[1] for t in test_data_fnames
                       if t[0] == fold][0]

    train = pd.read_csv(train_data_fname)
    test = pd.read_csv(test_data_fname)
    return train, test


def load_qry_codes(io_config):
    qry_codes_fname = io_config['file']['qry-codes']
    qry_codes = np.load(qry_codes_fname)

    return qry_codes


def load_mod(io_config):
    fold = io_config['fold']

    mod_fnames = io_config['file']['load-mod']
    mod_fname = [t[1] for t in mod_fnames
                 if t[0] == fold][0]

    with open(mod_fname, 'rb') as f:
        mod = pkl.load(f)

    msg = """
    Loading model from disk: {}
    """.format(mod_fname)
    debug_print(msg, V=VERBOSITY)

    ind_time = mod.s['model_data']['ind_time']
    ind_time = round(ind_time, PRECISION)

    return mod, ind_time


# Actions
def induction(train, mercs_config):
    model = mercs.MERCS()
    model.fit(train, **mercs_config, delimiter='.')

    ind_time = model.s['model_data']['ind_time']
    ind_time = round(ind_time, PRECISION)

    return model, ind_time


def inference_and_evaluation(model,
                             test_data,
                             qry_codes,
                             mercs_config,
                             eval_config):
    # Extract config
    _, q_targ, _ = codes_to_query(qry_codes)
    eval_kinds = eval_config.get('kinds', ['macro_f1'])

    msg = """
    run_mercs.py
    eval_kinds: {}
    """.format(qry_codes, eval_kinds)
    debug_print(msg, V=VERBOSITY)

    # Initialize
    head_tuple = ('q_idx', *eval_kinds)
    data_tuple = tuple([0] + [np.nan] * len(eval_kinds))
    tuple_list = [data_tuple] * len(qry_codes)

    inf_timing = np.zeros(len(qry_codes))
    # Actions
    for q_idx, q_code in enumerate(qry_codes):
        mercs_config['qry.code'] = q_code

        true_data = test_data.values[:, q_targ[q_idx]]
        pred_data = model.predict(test_data,
                                  **mercs_config,
                                  delimiter='.')

        evals = [round(eval_dict[kind](true_data, pred_data), PRECISION)
                 for kind in eval_kinds]
        tuple_list[q_idx] = q_idx, *evals
        inf_timing[q_idx] = round(model.s['model_data']['inf_time'], PRECISION)

        del pred_data

    results_df = pd.DataFrame.from_records(tuple_list, columns=head_tuple)

    return results_df, inf_timing


def tidy_timings(ind_timing, inf_timing):
    assert isinstance(ind_timing, (int,float))
    assert isinstance(inf_timing, np.ndarray)

    df = pd.DataFrame(inf_timing, columns=['inf_time'])
    df['ind_time'] = ind_timing
    df['q_idx'] = df.index

    df = df[['q_idx', 'ind_time', 'inf_time']]
    return df


# Outputs
def save_results(results_df, io_config):
    fold = io_config['fold']
    results_fnames = io_config['file']['results']

    results_fname = [t[1] for t in results_fnames
                     if t[0] == fold][0]

    results_df.to_csv(results_fname, index=False)

    return


def save_timings(timings_df, io_config):
    fold = io_config['fold']
    timings_fnames = io_config['file']['timings']

    timings_fname = [t[1] for t in timings_fnames
                     if t[0] == fold][0]

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


eval_dict = {'macro_f1': macro_f1,
             'micro_f1': micro_f1,
             'weigh_f1': weigh_f1,
             'accuracy': accuracy}      # Dict instead of entire class, is OK

# For executable script
if __name__ == '__main__':
    config_fname_outer_scope = sys.argv[1]
    fold_outer_scope = int(sys.argv[2])

    main(config_fname_outer_scope, fold_outer_scope)
