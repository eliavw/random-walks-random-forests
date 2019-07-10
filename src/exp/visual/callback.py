from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import warnings

from itertools import chain
import numpy as np


def extract_menu_inputs_menu_names_from_layout(layout):
    """
    Extract menu inputs and names from the layout of the Dash app.

    This extraction is useful because we automatically want to generate
    the relevant menus in different contexts.

    Parameters
    ----------
    layout:
        Layout of Dash app.

    Returns
    -------

    """

    menus = list(chain.from_iterable(_unfold(layout)))

    menu_names = [e.id for e in menus]
    menu_inputs = [Input(n, "value") for n in menu_names]

    return menu_inputs, menu_names


def filter_dataframe(df, column, values, kind=None):
    """
    Filter a given DataFrame df.

    This entails two things:
        1.  Select a 'column'
        2.  Delete every row in the DataFrame for which the column 'column' does
            not contain a value that is present in 'values'.

    Parameters
    ----------
    df: pd.DataFrame
        Input-dataframe which is filtered
    column: str
        Column name of the column that is scanned for values.
    values: {list, str, int, bool, float, ...}
        Values that have to be present in column
    kind: {None, 'range', 'pass', 'IDC', 'idontcare'}
        Kind of check that needs to happen.

    Returns
    -------

    """

    if isinstance(values, (str, int, bool, float)):
        values = [values]

    if isinstance(values, type(None)):
        values = []

    column = [c for c in df.columns if column in c][0]  # Pick first suitable one

    if "IDC" in values:
        kind = "idontcare"

    if kind is None:
        df_filt = df[df[column].isin(values)]
    elif kind in {"timeout"}:
        df_filt = df

        def f(row):
            if row[column] < values[0]:
                return row["score"]
            else:
                return 0.5

        df_filt["score"] = df.apply(f, axis=1)
    elif kind in {"Slider"}:
        df_filt = df[df[column] >= values[0]]

    elif kind in {"RangeSlider"}:
        df_filt = df[df[column] >= values[0]]
        df_filt = df_filt[df_filt[column] <= values[1]]
    elif kind in {"pass", "IDC", "idontcare"}:
        df_filt = df
    else:
        msg = """
        Did not recognize kind:    {}
        Not doing any filtering
        """.format(
            kind
        )
        warnings.warn(msg)

        df_filt = df

    return df_filt


def _unfold(x):
    recognized_menus = (dcc.Dropdown, dcc.Slider, dcc.RangeSlider)

    if isinstance(x, html.Div):
        return _unfold(x.children)
    elif isinstance(x, list):
        return [_unfold(e) for e in x if _unfold(e) is not None]
    elif isinstance(x, recognized_menus):
        return x
    else:
        return None
