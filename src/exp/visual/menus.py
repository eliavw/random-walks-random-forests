import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import warnings

from ..utils.extra import debug_print

VERBOSITY = 0


# Dropdown menus
def generate_multi_button(value, label=None):
    """
    Generate a single button in a dropdown menu.

    This means one entry in a dropdown menu. Such a entry consists of two parts,

    :param value:       The underlying value of the button. This is accesible
                        on the backend, invisible to the user.
    :param label:       The label of the button, i.e. what is visible to the
                        user.
    :return:            Simple dictionary that summarizes these two properties.
    """

    if label is None:
        label = str(value)

    button = {"label": label, "value": value}
    return button


def generate_dropdown_menu(identifier, values, labels=None, default=-1, multi=True):
    """
    Generate a single dropdown menu.

    Parameters
    ----------
    identifier: str
        ID of the menu, this is an identifier that later on is used to refer to
        this exact menu as an element of the html page.
    values: list
        The value of the menu can be regarded as some sort of 'state' in which
        the menu will be once the label corresponding to this value has been
        selected.
    labels: list
        Labels corresponding to the values. This is optional because we can also
        just use the underlying values as their own labels.
    default:
        The default value of the menu, i.e.; the value that is selected before
        users take any action.
            - An integer
                Referring to the i-th button in the menu. AFTER SORTING!
            - A string
                Referring to the actual value that you wish to be default.
    multi: bool
        Flag that indicates whether or not we allow multiple selections.

    Returns
    -------

    """
    assert isinstance(values, (np.ndarray, list))

    if labels is not None:
        assert len(labels) == len(values)
        values_labels = zip(values, labels)
        buttons = [
            generate_multi_button(value, label=label) for value, label in values_labels
        ]
    else:
        try:
            # Sometimes, it is not possible to sort. E.g., mix of int/float and str
            values.sort()
        except TypeError:
            msg = """
            Could not sort this column. Typically because there is a mix
            of int/float and strings.
            """
            warnings.warn(msg)

        buttons = [generate_multi_button(value) for value in values]

    buttons = buttons + [generate_multi_button("IDC", label="IDC")]

    if isinstance(default, (int)):
        default_value = buttons[-1]["value"]
    elif isinstance(default, (str)):
        assert default in values
        default_value = default
    else:
        raise TypeError(
            "Default must be integer,"
            "referring to the index of the entry in the menu, or"
            "be an actual value."
        )

    parameters = {
        "id": identifier,
        "options": buttons,
        "value": default_value,
        "multi": multi,
    }

    title = html.Label(identifier)
    menu = dcc.Dropdown(**parameters)

    return [title, menu]


def generate_dropdown_menus_from_df(df, relevant_columns=None, ignore_columns=None):
    """
    Generate a list of dropdown menus from a pandas DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe where each column contains a parameter that was varied.
    relevant_columns: list(str)
        List of column names that should be converted into a dropdown menu.
    ignore_columns: list(str)
        List of column names that should NOT be converted into a dropdown menu.

    Returns
    -------

    """

    if relevant_columns is None:
        relevant_columns = df.columns.tolist()

    if ignore_columns is not None:
        assert isinstance(ignore_columns, (list, tuple))
        assert isinstance(ignore_columns[0], str)
        relevant_columns = [e for e in relevant_columns if e not in ignore_columns]
        assert len(relevant_columns) > 0

    menus = []
    for col in relevant_columns:
        msg = """
        Building menu for parameter: {}
        """.format(
            col
        )
        debug_print(msg, V=VERBOSITY)

        identifier = col
        identifier = identifier.replace(".", "-")
        u_values = df[col].unique()

        if not isinstance(u_values, np.ndarray):
            u_values = np.array(u_values)

        msg = """
        u_values: {}
        """.format(
            u_values
        )
        debug_print(msg, V=VERBOSITY)

        menu = generate_dropdown_menu(identifier, u_values)
        menus.extend(menu)
    return menus


# Slider menus
def generate_slider_menu(
    identifier, min_value=None, max_value=None, default_value=None, step=None, kind=None
):
    """
    Generate one slider menu
    """

    def format_mark(number):
        return "{:0.2f}".format(number).rstrip("0") + "0"[0 : (number % 1 == 0)]

    if kind is None:
        kind = "default"

    title = html.Label(identifier)
    parameters = {"id": identifier}

    if kind in {"default"}:
        if min_value is None:
            min_value = 0
        if max_value is None:
            max_value = 100
        if default_value is None:
            default_value = 0
        if step is None:
            step = 1

        parameters = {
            **parameters,
            "min": min_value,
            "max": max_value,
            "step": step,
            "marks": {
                k: format_mark(k) for k in np.linspace(min_value, max_value, num=11)
            },
            "value": default_value,
        }

        slider = dcc.Slider(**parameters)
    elif kind in {"range", "range_slider"}:
        if min_value is None:
            min_value = 0
        if max_value is None:
            max_value = 100
        if step is None:
            step = 1

        marks = {k: format_mark(k) for k in np.linspace(min_value, max_value, num=11)}

        parameters = {
            **parameters,
            "min": min_value,
            "max": max_value,
            "step": step,
            "marks": {
                k: format_mark(k) for k in np.linspace(min_value, max_value, num=11)
            },
            "value": [min_value, max_value],
        }
        slider = dcc.RangeSlider(**parameters)
    elif kind in {"log", "logslider"}:
        if min_value is None:
            min_value = -3
        if max_value is None:
            max_value = 7
        if default_value is None:
            default_value = 2
        if step is None:
            step = 0.01

        parameters = {
            **parameters,
            "min": min_value,
            "max": max_value,
            "step": step,
            "marks": {
                k: format_mark(k) for k in np.linspace(min_value, max_value, num=11)
            },
            "value": default_value,
        }

        slider = dcc.Slider(**parameters)
    else:
        msg = """
        Did not recognize kind: {}
        """.format(
            kind
        )
        raise ValueError(msg)

    return [title, slider]
