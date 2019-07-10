from inspect import signature


def filter_kwargs_prefix(prefix, delimiter='.', **kwargs):
    """
    Only retain entries in kwargs that match the prefix.

    From the original key, we only retain what is left behind the prefix.

    Parameters
    ----------
    prefix
    delimiter
    kwargs

    Returns
    -------

    """

    prefix += delimiter

    relevant_kwargs = {k.split(prefix)[1]:v
                       for k,v in kwargs.items()
                       if prefix in k}

    return relevant_kwargs


def enter_value_in_dict_through_keychain(d, keychain, value):

    k = keychain[0]

    if len(keychain ) == 1:
        d[k] = value
    elif k not in d:
        d[k] = {}
        d[k] = enter_value_in_dict_through_keychain(d[k], keychain[1:], value)
    elif isinstance(d[k], dict):
        d[k] = enter_value_in_dict_through_keychain(d[k], keychain[1:], value)
    else:
        # Nothing to be done
        pass

    return d


def add_hierarchical_kwargs_to_dict(d, delimiter='.', **kwargs):
    keychains_values = [(k.split(delimiter), v) for k, v in kwargs.items()]

    for keychain, value in keychains_values:
        d = enter_value_in_dict_through_keychain(d, keychain, value)

    return d


def filter_kwargs_function(f, **kwargs):
    """
    From kwargs, extract the keywords that are arguments for function f.

    Parameters
    ----------
    f
    kwargs

    Returns
    -------

    """
    keywords_f = _extract_keywords(f)
    keywords_u = set(kwargs.keys())

    relevant_keywords = keywords_f.intersection(keywords_u)

    relevant_kwargs = {k: v for k,v in kwargs.items()
                       if k in relevant_keywords}
    return relevant_kwargs


def _extract_keywords(f):
    parameters = signature(f).parameters

    keywords = set(parameters.keys())
    return keywords
