"""
Collect and aggregate outputs
"""
import numpy as np
import pandas as pd
import json

from ..query.encoding import encode_attribute

# TODO: THIS IS HARDCODED!
TARG_ENCODING = encode_attribute(1,[0],[1])
MISS_ENCODING = encode_attribute(2,[0],[1])


# Aggregation
def aggregate_outputs(df_fns, kind='results'):
    """
    Collect in such a way that no single file is accessed more than once.
    """
    df = pd.DataFrame()

    column, uniq_fnames = extract_unique_fnames_kind(df_fns, kind=kind)

    for fn in uniq_fnames:
        try:
            if kind in {'results', 'timings'}:
                single_df = pd.read_csv(fn)             # Reading csv
            elif kind in {'qry_codes'}:
                q_codes = np.load(fn)                   # Reading npy
                single_df = transform_q_codes(q_codes)  # Transformation
            elif kind in {'mod_config'}:
                with open(fn, 'r') as f:
                    cfg = json.load(f)                  # Reading json

                single_df = transform_cfg(cfg)          # Transformation
            else:
                msg = """
                Did not recognize kind:\t{}
                """.format(kind)
                raise ValueError(msg)

            # Concatenation
            head_tuple = ('idx', 'f_idx')
            filt_df_fn = df_fns[df_fns[column] == fn]
            filt_df_fn = filt_df_fn[list(head_tuple)]

            """
            Now, whenever the same file appears for a combination of idx+f_idx,
            we add it to the aggregated dataframe. This uniqueness avoids 
            aggregating qry_codes and mod_config for each query in case of 
            parallel execution.
            
            Since results and timings are saved in seperate files in the case
            of parallel execution, this will still work then.
            """
            tuple_list = list(filt_df_fn.itertuples(index=False, name=None))
            uniq_tuples = set(tuple_list)
            for idx, f_idx in uniq_tuples:
                tmp = add_multiindex(single_df, idx, f_idx)
                df = pd.concat([df, tmp], sort=False)

        except FileNotFoundError as e:
            msg = """
            A FileNotFoundError occurred, probably because this Idx failed:     {}
            """.format(e.args[-1])
            print(msg)

    return df


# Transformations
def transform_q_codes(q_codes):
    df = pd.DataFrame()
    targ_encoding = TARG_ENCODING
    miss_encoding = MISS_ENCODING
    nb_qrys, nb_atts = q_codes.shape

    df['targ'] = extract_attributes(q_codes, targ_encoding)
    df['targ'] = df['targ'].astype('category')
    df['t_idx'] = df['targ'].cat.codes

    df['perc_miss'] = count_attributes(q_codes, miss_encoding) / nb_atts * 100
    df['q_idx'] = np.arange(nb_qrys)

    return df


def transform_cfg(cfg):
    # Prelims
    child = cfg['child']
    dataset = cfg['dataset']

    cfg_fit_and_predict = cfg[child]

    # If the model came from disk, we collect that info as well.
    load = cfg['io']['file'].get('load-mod-config', False)
    if load:
        msg = """
        Model was loaded from disk: {}
        """.format(cfg['io']['file']['load-mod-config'])
        print(msg)
        with open(cfg['io']['file']['load-mod-config'], 'r') as f:
            cfg_disk = json.load(f)

        mod_cfg = cfg_disk.get("mod", {})
        fit_cfg = cfg_disk.get("fit", {})

        fit_cfg = {'fit.' + k: v for k,v in fit_cfg.items()}
        mod_cfg = {'mod.' + k: v for k,v in mod_cfg.items()}

        cfg_fit_and_predict = {**cfg_fit_and_predict,
                               **fit_cfg,
                               **mod_cfg}

    # Actual transformation

    head_tuple = ('dataset',
                  *cfg_fit_and_predict.keys())
    data_tuple = (dataset,
                  *cfg_fit_and_predict.values())

    df = pd.DataFrame.from_records([data_tuple], columns=head_tuple)

    return df


# Helpers - Transformations
def count_attributes(q_codes, encoding):
    """
    Count the number of appearances of a certain value, row-wise.
    """
    return np.count_nonzero(q_codes == encoding, axis=1)


def extract_attributes(q_codes, encoding):
    """
    Extract the attributes that fulfill a certain role, row-wise.
    """
    nb_queries = q_codes.shape[0]
    encod_atts = np.transpose(np.nonzero(q_codes == encoding))

    d = [[] for row in range(nb_queries)]
    for qry_idx, att_idx in encod_atts:
        d[qry_idx].append(att_idx)

    d = [tuple(l) for l in d]
    return d


# Helpers - Aggregation
def extract_unique_fnames_kind(df, kind=None):
    """
    Extract unique fnames from df of outputs.

    This allows us to read every file just once.
    """
    column = [c for c in df.columns if kind in c][0]
    uniq_fnames = df[column].unique()
    return column, uniq_fnames


def add_multiindex(df, idx, f_idx):
    """
    Add multi-index correctly.
    """
    df['idx'] = idx
    df['f_idx'] = f_idx

    if 'q_idx' in df.columns:
        df = df.set_index(['idx', 'f_idx', 'q_idx'])
    else:
        df = df.set_index(['idx', 'f_idx'])
    return df
