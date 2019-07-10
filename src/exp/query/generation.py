import numpy as np
from ..utils.extra import filter_kwargs_to_signature
from .encoding import encode_attribute


def compile_queries(mode='basic', nb_atts=None, random_seed=997, **kwargs):
    """
    Compile queries

    A query specifies a prediction task that the model has to solve.

    Parameters
    ----------
    mode: str
        Query generation method.
    nb_atts: int
        Number of attributes of the dataset.
    random_seed: int
        Random seed passed to the generation algorithms that rely on randomization.
    kwargs: dict
        Optional keyword arguments for specific query generation strategies.

    Returns
    -------

    """

    # Prelims
    if nb_atts is None or (nb_atts < 2):
        msg = """
        Nb of atts provided: {} is not an allowed value
        """.format(nb_atts)
        raise ValueError(msg)
    else:
        atts = list(range(nb_atts))

    induction_desc, induction_targ = atts, atts

    # Query Compilation
    if mode in {'basic', 'default'}:
        q_desc, q_targ, q_miss = basic_query_algo(atts)
    elif mode in {'iterative', 'it_missing_2'}:
        kwargs["random_seed"] = random_seed
        red_kwargs = filter_kwargs_to_signature(iterative_query_algo, kwargs)
        q_desc, q_targ, q_miss = iterative_query_algo(atts, **red_kwargs)
    else:
        msg = """
        Did not recognize keyword: {}
        Accepted keywords are: ['basic', 'iterative']
        """.format(mode)
        raise ValueError(msg)

    return induction_desc, induction_targ, q_desc, q_targ, q_miss


def basic_query_algo(atts):
    """
    Leave-one-out prediction, with a maximum of 50 queries.

    :param atts:
    :return:
    """

    query_targ_sets = [[i] for i in atts if i < 50]

    # All the remaining atts are used as inputs
    query_desc_sets = [list(set(atts) - set(qts)) for qts in query_targ_sets]
    query_miss_sets = [[] for i in query_desc_sets]
    return query_desc_sets, query_targ_sets, query_miss_sets


def iterative_query_algo(atts,
                         nb_steps=10,
                         nb_diff_configs=50,
                         random_seed=997):
    """
    We randomly select some attributes which will be targets.

    Each target is predicted in one query, all the rest is descriptive. Then we gradually leave out descriptive attributes.
    """

    np.random.seed(random_seed)
    nb_atts = len(atts)

    query_desc_sets, query_targ_sets, query_miss_sets = [], [], []

    shuffled_atts = list(np.random.permutation(atts).tolist() for i in range(nb_diff_configs))

    orig_code = generate_query_code(nb_atts, miss_size=0)
    ms_steps = generate_ms_steps(orig_code, nb_steps)  # Query param determines the amount of steps.
    code = orig_code.copy()

    for atts_shuffle in shuffled_atts:

        # Getting the query, and saving it
        desc, targ, miss = code_to_query(code, atts_shuffle)  # Generate the actual query (3 arrays)
        query_desc_sets.append(desc), query_targ_sets.append(targ), query_miss_sets.append(miss)

        for step in ms_steps:
            code = desc_to_miss(code, step)  # Update code

            # Getting the query, and saving it
            desc, targ, miss = code_to_query(code, atts_shuffle)
            query_desc_sets.append(desc), query_targ_sets.append(targ), query_miss_sets.append(miss)

        code = orig_code.copy()  # Do not forget to reset the code to its original form.

    return query_desc_sets, query_targ_sets, query_miss_sets


# Helpers
def generate_query_code(nb_atts,
                        desc_size=None,
                        targ_size=1,
                        miss_size=None):
    """
    Generate an array of the same length as the original attribute array.

    The query code means the following:
        0:  Descriptive attribute
        1:  Target attribute
        -1: Missing attribute

    TODO: Switch to more general code + Centralize all coding affairs.
    """

    desc_encoding = encode_attribute(2,[2],[3])
    targ_encoding = encode_attribute(3,[2],[3])
    miss_encoding = encode_attribute(1,[2],[3])

    # Some automatic procedures if we are given some freedom.
    if desc_size == None and miss_size == None:
        desc_size = nb_atts - targ_size  # All the rest is assumed descriptive
        miss_size = 0
    elif desc_size != None and miss_size == None:
        miss_size = nb_atts - targ_size - desc_size
    elif desc_size == None and miss_size != None:
        desc_size = nb_atts - targ_size - miss_size
    else:
        pass

    # Bad user choices can still cause meaningless queries, so we check
    assert ((desc_size + targ_size + miss_size) == nb_atts)
    assert (targ_size > 0)
    assert (desc_size > 0)

    # The actual encoding
    code = [desc_encoding for i in range(desc_size)]
    code.extend([targ_encoding for i in range(targ_size)])
    code.extend([miss_encoding for i in range(miss_size)])
    return code


def generate_ms_steps(code, param):
    """
    Generate an array that contains the (integer) steps in which we increase the set of missing attributes.

    This is in the context of converting descriptive attributes to missing ones.

    :param param -  If parameter is a float < 1, it is interpreted as the percentage increase that is desired
                    If parameter is a int > 1, it is interpreted as the amount of attributes that needs to be added
                    to the missing set at each step.
    :param code - The starting code
    """

    max_amount_steps = int(1 / param) if param < 1 else int(param)
    available_atts = code.count(0)

    ms_sizes = np.linspace(0, available_atts, num=max_amount_steps + 1, dtype=int).tolist()
    ms_steps = [ms_sizes[i] - ms_sizes[i - 1]
                for i in range(1, len(ms_sizes))
                if (ms_sizes[i] - ms_sizes[i - 1] > 0)]

    return ms_steps[:-1] # We do not return the last entry, since that means everything missing.


def desc_to_miss(code, amount=1):
    """
    Change the first 'amount' of 0's to -1's
    """
    desc_encoding = encode_attribute(2, [2], [3])
    miss_encoding = encode_attribute(1, [2], [3])


    changes, i = 0, 0

    while i < len(code) and changes < amount:
        if code[i] == desc_encoding:
            code[i] = miss_encoding
            changes += 1
        else:
            pass
        i += 1
    return code


def code_to_query(code, atts=None):
    """
    Change the code-array to an actual query, which are three arrays.

    :param code:                Array that contains:
                                     0 for desc attribute
                                     1 for target attribute
                                    -1 for missing attribute
    :param atts:                Array that contains the attributes (indices)
    :return: Three arrays.      One for desc atts indices, one for targets,
                                one for missing

    TODO(elia): The coding strategy is still hardcoded here. Fix this.
    """

    desc_encoding = encode_attribute(2,[2],[3])
    targ_encoding = encode_attribute(3,[2],[3])
    miss_encoding = encode_attribute(1,[2],[3])

    if atts is None: atts = list(range(len(code)))
    assert len(code) == len(atts)

    desc = [x for i, x in enumerate(atts) if code[i] == desc_encoding]
    targ = [x for i, x in enumerate(atts) if code[i] == targ_encoding]
    miss = [x for i, x in enumerate(atts) if code[i] == miss_encoding]
    return desc, targ, miss