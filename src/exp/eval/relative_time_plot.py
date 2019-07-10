from .utils import build_f_dict, insert_name_column


def build_relative_time_df(df_tmg,
                           df_cfg,
                           baseline_filter,
                           baseline_column='inf_time',
                           group_on=None):

    # T1: Average Inf Times across f_idx and q_idx
    f_dict = build_f_dict(df_tmg)
    df_tmg = df_tmg.groupby(level=['idx']).agg(f_dict)

    # T2: Join config and timing results
    join_idx_names = ['idx']
    df_1 = df_tmg.join(df_cfg, how='inner', on=join_idx_names)

    # T3: Filter baseline
    df_baseline = filter_baseline(df_1, baseline_filter)
    dict_baseline = collect_values(df_baseline, key='dataset', baseline_column=baseline_column)
    df_2 = insert_baseline_column(df_1, dict_baseline, key='dataset', original_column_name=baseline_column)

    # T4: Insert relative baseline column
    df_3 = insert_relative_column(df_2, original_column_name=baseline_column)

    # T5: Insert name column
    df_4 = insert_name_column(df_3, include_columns=group_on)

    # Index again
    idx_names = ['idx', 'name']
    df_4.reset_index(inplace=True)
    df_4.set_index(idx_names, inplace=True)
    df_4.sort_index(inplace=True)

    return df_4


def filter_baseline(df, baseline):
    """
    Only keep the rows of the DataFrame that meet certain conditions.

    These conditions are specified in baseline. Cf. parameters description for
    clarity.

    Parameters
    ----------
    df: pd.DataFrame
    baseline: list(tuple)
        List of tuples (column name. value). Each tuple basically defines a
        test for which the row of the dataframe has to succeed in order not be
        filtered.
            E.g. baseline = [('predict.algo', 'IT'), ('predict.its', 2)]
                A row must have 'predict.algo' == 'IT' AND 'predict.its' == 2
                in order to survive the filter.

    Returns
    -------

    """
    assert isinstance(baseline, list)
    assert isinstance(baseline[0], tuple)

    def f(row):
        for k,v in baseline:
            if row[k] != v: return False
        return True

    return df[df.apply(f, axis=1)]


def collect_values(df, key='dataset', baseline_column='inf_time'):
    df_filt = df[[key, baseline_column]]

    arr = df_filt.values
    keys = arr[:, 0]
    vals = arr[:, 1]

    d = {k: v for k, v in zip(keys, vals)}

    return d


def insert_baseline_column(df, dict_baseline, key='dataset', original_column_name='inf_time'):

    def f(row):
        return dict_baseline[row[key]]

    baseline_column_name = derived_column_name(original_column_name, prefix='baseline')

    df[baseline_column_name] = df.apply(f, axis=1)
    return df


def insert_relative_column(df, original_column_name='inf_time', kind='baseline'):
    baseline_column_name = derived_column_name(original_column_name, prefix=kind)

    def f(row):
        return row[original_column_name] / row[baseline_column_name]

    relative_column_name = derived_column_name(original_column_name, prefix='relative')

    df[relative_column_name] = df.apply(f, axis=1)

    return df


def derived_column_name(original_column_name, prefix='baseline', delimiter='_'):
    return prefix + delimiter + original_column_name
