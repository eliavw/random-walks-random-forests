from inspect import signature
import pandas as pd


def pretty_print_dict(dictionary, prefix=""):
    """
    Decently formatted print of nested dictionaries.

    Sometimes this is nice, especially in notebooks.

    TODO(elia): Alignment could be better.
    """
    for k, v in dictionary.items():
        new_prefix = prefix + "\t"
        if isinstance(v, dict):
            print(prefix + k)
            pretty_print_dict(v, prefix=new_prefix)
        else:
            print(prefix + "{} \t {}".format(k, v))
    print("\n")
    return


def debug_print(msg, level=1, V=0, **kwargs):

    """
    def get_var_name(var):
        var_name = [k for k, v in locals().items() if v is var][0]
        return var_name

    if kwargs.keys() > {'msg', 'level', 'V'}:
        print('INSIDE')
        relevant = {k:v for k,v in kwargs.items()
                    if k not in {'msg', 'level', 'V'}}
        for k,v in relevant.items():
            msg+="k: {}".format(v)
    """

    if V >= level:
        print(msg+"\n")
    return


def generate_keychain(entries, sep='-'):
    """
    Generate a keychain from entries.

    A keychain is a structured string. This is something that happens
    at various points in the codebase to have semi-structured, yet
    human-readable strings.

    So, this is basically a tiny method to ensure consistency throughout the
    project.

    Parameters
    ----------
    entries
    sep

    Returns
    -------

    """
    assert isinstance(entries, list)

    keychain = entries[0]
    for e in entries[1:]:
        keychain += sep+e
    return keychain


def mem_usage(pandas_obj, scale='kilo'):
    """
    Memory usage of a pandas object

    Parameters
    ----------
    pandas_obj: {pd.DataFrame, pd.Series}

    Returns
    -------
    result: str
        Pretty-print string that contains the size of the pandas object
        in megabytes.
    """

    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    elif isinstance(pandas_obj, pd.Series):
        usage_b = pandas_obj.memory_usage(deep=True)
    else: # we assume if not a df it's a series
        msg = """
        Only takes pd.DataFrame or pd.Series as input.
        Type: {} not recognized.
        """.format(type(pandas_obj))
        raise ValueError(msg)

    oom = {'kilo':  1,
           'mega':  2,
           'giga':  3}

    memory_usage = usage_b / 1024 ** oom[scale]  # convert bytes to kilobytes
    msg = """
    {:03.2f} {}B
    """.format(memory_usage, scale)

    print(msg)
    return


def filter_kwargs_to_signature(f, kwargs):
    assert callable(f)
    assert isinstance(kwargs, dict)

    sig = signature(f)

    # intersection of keys
    joint_keys = set(sig.parameters.keys()) & set(kwargs.keys())

    return {k:kwargs[k] for k in joint_keys}
