import numpy as np


def code_to_query(code, attributes=None):
    """
    Split the code array into three arrays of attributes (desc, targ and miss)

    Args:
        code:           np.ndarray, shape (nb_attributes, )
                        One-dimensional numpy array that encodes a query. Each entry
                        encodes the function of the associated attribute
        attributes:     np.ndarray, shape (nb_attributes, ), default=None
                        Numpy array that contains the indices of the attributes whose
                        function is encoded in the codes. If None, we assume that the
                        attributes indices are simply np.arange(nb_attributes)

    Returns:

    """

    assert isinstance(code, np.ndarray)
    assert code.ndim == 1

    nb_atts = code.shape[0]
    if attributes is None:
        attributes = np.arange(nb_atts)
    assert isinstance(attributes, np.ndarray)
    assert code.shape == attributes.shape

    desc_encoding = encode_attribute(0, [0], [1])
    targ_encoding = encode_attribute(1, [0], [1])
    miss_encoding = encode_attribute(2, [0], [1])

    desc, targ, miss = [], [], []

    for i, x in enumerate(attributes):
        if code[i] == desc_encoding:
            desc.append(x)
        elif code[i] == targ_encoding:
            targ.append(x)
        elif code[i] == miss_encoding:
            miss.append(x)
        else:
            msg = """
            Did not recognize encoding: {}\n
            This occured in code: {}\n
            Ignoring this entry.
            """.format(
                code[i], code
            )
            raise ValueError(msg)

    return desc, targ, miss


def query_to_code(q_desc, q_targ, q_miss=None, atts=None):
    if atts is None:
        atts = determine_atts(q_desc, q_targ, q_miss)

    code = [encode_attribute(a, q_desc, q_targ) for a in atts]

    return np.array(code)


def determine_atts(desc, targ, miss):
    """
    Determine the entire list of attributes.
    """
    atts = list(set(desc).union(targ).union(miss))
    atts.sort()
    return atts


def encode_attribute(att, desc, targ):
    """
    Encode the 'role' of an attribute in a model.

    `Role` means:
        - Descriptive attribute (input)
        - Target attribute (output)
        - Missing attribute (not relevant to the model)
    """

    check_desc = att in desc
    check_targ = att in targ

    code_int = check_targ * 2 + check_desc - 1

    return code_int
