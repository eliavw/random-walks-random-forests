import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..utils import code_to_query, debug_print

VERBOSITY = 0


def base_induction_algorithm(
    data,
    m_codes,
    metadata,
    classifier=DecisionTreeClassifier,
    regressor=DecisionTreeRegressor,
    classifier_kwargs=None,
    regressor_kwargs=None,
):
    """
    Basic induction algorithm. Models according to the m_codes it receives.

    Parameters
    ----------
    data:                   np.ndarray
    m_codes:                np.ndarray
    metadata:               dict
    classifier:
    regressor:
    classifier_kwargs:      dict
    regressor_kwargs:       dict

    Returns
    -------

    """
    assert isinstance(data, np.ndarray)

    # Init
    if classifier_kwargs is None:
        classifier_kwargs = dict()
    if regressor_kwargs is None:
        regressor_kwargs = dict()

    n_rows, n_cols = data.shape
    m_list = []

    attributes = set(range(n_cols))
    nominal_attributes = metadata.get("nominal_attributes")
    numeric_attributes = metadata.get("numeric_attributes")
    msg = """Not all attributes of m_codes are accounted for in metadata"""
    assert not (attributes - nominal_attributes - numeric_attributes), msg

    # Codes to queries
    ids = [(d, t) for d, t, x in [code_to_query(m_code) for m_code in m_codes]]

    for desc_ids, targ_ids in ids:
        msg = """
        Learning model with desc ids:    {}
                            targ ids:    {}
        """.format(
            desc_ids, targ_ids
        )
        debug_print(msg, level=1, V=VERBOSITY)

        if set(targ_ids).issubset(nominal_attributes):
            kwargs = classifier_kwargs
            learner = classifier
            out_kind = "nominal"
        elif set(targ_ids).issubset(numeric_attributes):
            kwargs = regressor_kwargs
            learner = regressor
            out_kind = "numeric"
        else:
            msg = """
            Cannot learn mixed (nominal/numeric) models
            """
            raise ValueError(msg)

        # Learn a model for current desc_ids-targ_ids combo
        m = _learn_model(data, desc_ids, targ_ids, learner, out_kind, **kwargs)
        m_list.append(m)

    return m_list


# Random edit


def _learn_model(data, desc_ids, targ_ids, learner, out_kind="numeric", **kwargs):
    """
    Learn a model from the data.

    The arguments of this function determine specifics on which task,
    which learner etc.

    Model is a machine learning method that has a .fit() method.

    Parameters
    ----------
    data:           np.ndarray
                    The actual input data, all columns
    desc_ids:       set
                    Set of integers identifying the descriptive columns
    targ_ids:       set
                    Set of integers identifying the target columns
    learner:        algorithm
                    Algorithm used for learning a ML-model
    kwargs:         dict
                    Keyword arguments that are passed to ML-algorithm constructor

    Returns
    -------

    """

    i, o = data[:, desc_ids], data[:, targ_ids]

    if i.ndim == 1:
        # We always want 2D inputs
        i = i.reshape(-1, 1)
    if o.shape[1] == 1:
        # If output is single variable, we need 1D matrix
        o = o.ravel()

    try:
        model = learner(**kwargs)
        model.fit(i, o)
    except ValueError as e:
        print(e)

    # Bookkeeping
    model.desc_ids = desc_ids
    model.targ_ids = targ_ids
    model.out_kind = out_kind
    return model
