import numpy as np
import pandas as pd

from .utils import build_f_dict, insert_name_column
from ..utils.extra import (debug_print)

VERBOSITY = 0


# Preprocess
def preprocess_aggr_df(df, kind='df_res', include_columns=None):
    """
    Apply the appropriate series of transformations on the initial
    dataframe to get it in appropriate shape.

    Parameters
    ----------
    df
    kind

    Returns
    -------

    """

    # T0
    df = insert_category_dtype(df)

    if 'res' in kind:
        # T1
        f_dict = build_f_dict(df)
        df = df.groupby(level=['idx', 'q_idx']).agg(f_dict)

    elif 'qry' in kind:
        # T1
        f_dict = build_f_dict(df)
        df = df.groupby(level=['idx', 'q_idx']).agg(f_dict)
        df = insert_category_dtype(df)

        # T2
        #df = insert_targ_idx_column(df)

    elif 'cfg' in kind:
        # TODO: This is hardcoded
        if 'pred.type' not in df.columns:
            if 'inference_algorithm' in df.columns:
                df = df.rename(columns={'inference_algorithm': 'pred.type'})

        # T1
        f_dict = build_f_dict(df)
        df = df.groupby(level=['idx']).agg(f_dict)

        # T3
        df = insert_name_column(df, include_columns=include_columns)

        # T0 again
        df = insert_category_dtype(df)

        # T5
        #nan_value = 'PGM'
        #df['pred.type'] = df['pred.type'].cat.add_categories([nan_value]).fillna(nan_value)

    elif 'tmg' in kind:
        # T1
        f_dict = build_f_dict(df)
        df = df.groupby(level=['idx', 'q_idx']).agg(f_dict)

    return df


def build_tmg_histogram(df_tmg, df_cfg):
    # Filter df_tmg, df_cfg
    df_cfg_f = df_cfg[['dataset', 'pred.type']]

    # Reset Index df_tmg
    idx_names = ['idx']
    df_tmg_f = df_tmg.reset_index()
    df_tmg_f.set_index(idx_names, inplace=True)

    # Transformation: Add dataset column
    join_idx_names = ['idx']
    df_1 = df_tmg_f.join(df_cfg_f, how='inner', on=join_idx_names)

    # Transformation: Rename the 'pred.type' column to be the 'name'
    df_2 = df_1.rename(columns={'pred.type': 'name'})

    # Transformation: Change Index
    idx_names = ['idx', 'name']
    df_2.reset_index(inplace=True)
    df_2.set_index(idx_names, inplace=True)
    df_2.sort_index(inplace=True)

    return df_2


def build_df_plt(df_res, df_qry, df_cfg, df_tmg, score='macro_f1'):
    # Choose performance metric
    df_res.rename(columns={score: 'score'}, inplace=True)

    # Filter df_qry, df_cfg
    df_qry_f = df_qry[['t_idx', 'perc_miss']]
    df_cfg_f = df_cfg[['dataset', 'name']]
    idx_names = df_res.index.names

    # Join df_res, df_tmg
    join_idx_names = ['idx', 'q_idx']
    df_0 = df_res.join(df_tmg, how='inner', on=join_idx_names)
    df_0.index.set_names(idx_names, inplace=True)

    # Join df_0, df_qry
    join_idx_names = ['idx', 'q_idx']

    df_1 = df_0.join(df_qry_f, how='inner', on=join_idx_names)
    df_1.index.set_names(idx_names, inplace=True)

    # Add Base Performance
    df_1 = _insert_score_base(df_1)
    df_1 = _insert_score_rel(df_1)

    # Join df_1, df_cfg (i.e. add dataset column)
    idx_names = df_1.index.names
    join_idx_names = ['idx']

    df_2 = df_1.join(df_cfg_f, how='inner', on=join_idx_names)
    df_2.index.set_names(idx_names, inplace=True)

    # New Indices
    idx_names = ['idx', 'name', 'q_idx']
    df_2.reset_index(inplace=True)
    df_2.set_index(idx_names, inplace=True)
    df_2.sort_index(inplace=True)

    df_plt = df_2
    return df_plt


# Create Lineplot
def build_df_lineplot(df):
    """
    Create a DataFrame appropriate for a line plot.

    Convert the general-purpose df_plt to a special purpose dataframe for
    a line plot. This line plot depicts the rank across multiple difficulty
    levels. We use the percentage of missing attributes as a proxy for this
    difficulty level.

    We do some coarse rounding, to allow comparison across different datasets.

    Parameters
    ----------
    df: pd.DataFrame
        Output from build_df_plt.

    Returns
    -------

    """

    # Step 1: Drop unnecessary information
    df = df.drop(columns=['t_idx', 'score_base'])

    # Step 2: Rounding of entries
    df['perc_miss'] = df['perc_miss'].apply(lambda x: np.round(x, -1))
    # df['score'] = df['score'].apply(lambda x: np.round(x, 4)*100)

    # Compute Rank
    df = _insert_rank_columns(df, ranking_criterion='score')

    msg = """
    df columns after computing all the ranks: {}\n
    df index after adding all the ranks: {}
    """.format(df.columns.names, df.index.names)
    debug_print(msg, V=VERBOSITY)

    # Step 3: Average ranks and scores
    df = df.reset_index()
    df.set_index(['idx', 'name', 'q_idx', 'perc_miss'], inplace=True)

    # Mean rank and score across different q_idx with identical perc_miss
    result = pd.DataFrame()
    for quantity in {'score',
                     'score_rel',
                     'rank',
                     'aligned_rank',
                     'global_aligned_rank',
                     'global_dataset_aligned_rank',
                     'inf_time',
                     'ind_time'}:
        result[quantity] = df[quantity].mean(level=['idx', 'name', 'perc_miss'])

    msg = """
    result columns after computing all the ranks: {}
    result index after computing all the ranks: {}
    """.format(result.columns.names, result.index.names)
    debug_print(msg, V=VERBOSITY)

    # Mean rank and score across idxs
    result = result.mean(level=['name', 'perc_miss'])
    result = result.reset_index()

    # Tiny hack to get a multi-index. For compatibility with 'generate_trace_data'
    result['range_index'] = result.index
    result.set_index(['range_index', 'name'], inplace=True)
    return result


# ---  ---- ---  ----
# Transformations listed here
# ---  ---- ---  ----

# Transformation 0 - Optimize types of DF
def insert_category_dtype(df):
    """
    All non-numeric dtypes get converted to a category for saving memory.

    Parameters
    ----------
    df

    Returns
    -------

    """
    for col in df:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            df[col] = df[col].astype('category')
        else:
            pass
    return df


# Transformation 2 - Add t_idx column
def insert_targ_idx_column(df):
    df['t_idx'] = df.targ.cat.codes
    return df


# Transformation 4 - Add Base Performance Column
def _insert_score_base(df):
    def f(row):
        return row['score'] * 100 if row['perc_miss'] <= 0.01 else np.nan

    df['score_base'] = df.apply(f, axis=1) # TODO: Rename this entry!

    df.fillna(method='ffill', inplace=True)
    return df


def _insert_score_rel(df):
    # Adaptive function
    def f(row):
        return (row['score']*100)/row['score_base']

    df['score_rel'] = df.apply(f, axis=1)

    return df


# Transformation 5 - Add baseline avg query time
def _insert_inf_time_base(df, baseline=('pred.type', 'MI')):

    def collect_values(df, attribute, value):
        df_filt = df[df[attribute] == value][['dataset', 'inf_time']]

        arr = df_filt.values
        keys = arr[:, 0]
        vals = arr[:, 1]

        d = {k: v for k, v in zip(keys, vals)}

        return d

    base_inf_dict = collect_values(df, baseline[0], baseline[1])

    def f(row):
        if row[baseline[0]] == baseline[1]:
            return row['inf_time']
        else:
            return base_inf_dict[row['dataset']]

    df['inf_time_base'] = df.apply(f, axis=1)

    return df


def _insert_inf_time_relative(df):
    def f(row):
        return row['inf_time']/row['inf_time_base']

    df['inf_time_rel'] = df.apply(f, axis=1)
    return df


# Helpers - Build Lineplot
def _insert_rank_columns(df, ranking_criterion='score'):
    """
    Add a Rank column to the DataFrame.

    :param df:                  DataFrame formatted as the output of
                                `merge_df_perf_and_df_qry`-method.
                                Or at least sufficiently alike.

    :param ranking_criterion:   Criterion upon which the ranking is based.
    :return:
    """

    df = df.reset_index()
    df.set_index(['idx', 'name', 'q_idx', 'dataset', 'perc_miss'], inplace=True)

    df = _insert_aligned_score_column(df)

    rank = df.groupby(level=['q_idx', 'dataset'])[ranking_criterion].rank(ascending=False)  # Rank within q_idx, dataset
    df['rank'] = rank

    aligned_rank = df.groupby(level=['q_idx'])['aligned_score'].rank(ascending=False)  # Rank within q_idx HARDCODED!
    df['aligned_rank'] = aligned_rank

    global_aligned_rank = df.groupby(level=['perc_miss'])['aligned_score'].rank(ascending=False)  # Rank within perc_missing
    df['global_aligned_rank'] = global_aligned_rank

    global_dataset_aligned_rank = df.groupby(level=['dataset', 'perc_miss'])['aligned_score'].rank(ascending=False)  # Rank within perc_missing
    df['global_dataset_aligned_rank'] = global_dataset_aligned_rank

    df = df.reset_index()
    df.set_index(['idx', 'name', 'q_idx'], inplace=True)

    return df


def _insert_aligned_score_column(df):
    df = _add_avg_score(df)

    def f(row):
        return row['score'] - row['avg_score']

    df['aligned_score'] = df.apply(f, axis=1)

    return df


def _add_avg_score(df):

    avg_score = pd.DataFrame()
    avg_score['avg_score'] = df['score'].mean(level=['q_idx', 'dataset'])

    avg_score = avg_score.reset_index()
    df = df.reset_index()
    result = pd.merge(avg_score, df, on=['q_idx', 'dataset'])

    result.set_index(['idx', 'name', 'q_idx', 'dataset', 'perc_miss'], inplace=True)

    return result


