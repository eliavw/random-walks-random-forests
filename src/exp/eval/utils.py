import numpy as np
import pandas as pd

def build_f_dict(df):
    """
    Create a dict of functions

    This allows us to take the mean only in columns where it actually means
    something.

    Parameters
    ----------
    df

    Returns
    -------

    """
    f_dict = {col:  np.mean
                    if pd.api.types.is_numeric_dtype(df[col])
                    else 'first'
              for col in df}
    return f_dict


def insert_name_column(df,
                       include_columns=None,
                       ignore_columns=('dataset',),
                       **format_key_kwargs):
    """
    Insert the canonical name column in the dataframe
    """

    if include_columns is not None:
        ignore_columns = set(df.columns)-set(include_columns)
        ignore_columns = tuple(ignore_columns)

    df['name'] = df.apply(lambda x: _derive_name(x,
                                                 ignore_columns=ignore_columns,
                                                 **format_key_kwargs),
                          axis=1)
    return df


# Helpers - Transformation 3 - Add name column
def _format_key(k, retain=2, delimiter='.', **kwargs):
    l = k.split(delimiter)

    if len(l) > retain:
        l = l[-retain:]

    formatted_key = delimiter.join(l)

    return formatted_key


def _dict_to_str(d):
    s = str(d)

    for char in {"{", "}", "'"}:
        s = s.replace(char, "")

    for char in {", "}:
        s = s.replace(char, "|")

    for char in {": "}:
        s = s.replace(char, "=")

    return s


def _derive_name(row, ignore_columns=('dataset',), **format_key_kwargs):
    keys = row.index.values.tolist()
    vals = row.values.tolist()

    d = {k:v for k, v in zip(keys, vals)}

    for c in ignore_columns:
        d.pop(c, None)

    d = {_format_key(k, **format_key_kwargs): v
         for k, v in d.items()}

    name = _dict_to_str(d)
    return name
