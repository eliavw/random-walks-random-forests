import os
import re
import shutil
import warnings

from os.path import dirname


# Directories
def ensure_dir(d, empty=True):
    """
    Ensure that an (EMPTY) dir exists.

    Cf. https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
    """

    if os.path.exists(d):
        if empty:
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        else:
            pass
    else:
        os.makedirs(d, exist_ok=True)

    return


def make_dname(name=None, parent_dir=None, id=None, kind=None, flip=False):

    code_string = gen_appendix(id, kind) if id is not None else ''

    if name is None:
        # Default: root_dir
        utils_dir = dirname(__file__)
        exp_dir = dirname(utils_dir)
        src_dir = dirname(exp_dir)
        root_dir = dirname(src_dir)

        result = root_dir              # Default: parent directory of cwd
    else:
        if parent_dir is None:
            # Default: root_dir
            parent_dir = make_dname()

        result = os.path.join(parent_dir, name, code_string)

        if flip:
            result = os.path.join(parent_dir, code_string, name)

    return result


# Fname
def make_fname(name='filename', extension='json', dname=None):

    if dname is None:
        dname = make_dname()

    if extension != '' and extension is not None:
        if extension.startswith('.'):
            fname = name + extension
        else:
            fname = name+'.'+extension
    else:
        fname = name
        
    return os.path.join(dname, fname)


# Filenames
def gen_appendix(idx, kind=None):
    """
    Generate specific codestrings used in filesystem.


    Parameters
    ----------
    idx: int
        The number (int) that should be standardized
    kind: str
        The prefix

    Returns
    -------

    """

    if kind in {None, 'ID', 'id'}:
        appendix = str(idx).zfill(6)
    elif kind in {'F', 'f', 'Fold', 'fold'}:
        appendix = "F" + str(idx).zfill(2)
    elif kind in {'Q', 'q', 'query', 'Query', 'queries', 'Queries'}:
        appendix = "Q" + str(idx).zfill(4)
    elif kind in {'E', 'e', 'Exp', 'exp'}:
        appendix = "E" + str(idx).zfill(4)
    else:
        msg = """
        Did not recognize kind of appendix requested: '{}'\n
        Assuming default option of 'ID' instead.
        """.format(kind)
        warnings.warn(msg)
        appendix = gen_appendix(idx, kind='ID')

    return appendix


def extract_idx_from_string(string, int_only=True):
    # Define regex
    pattern_regex = '\_[A-Z]+\d+'
    kind_regex = '[A-Z]+'
    idx_regex = '\d+'

    # Extract regex
    patterns = re.findall(pattern_regex, string)

    if int_only:
        p = patterns[0]
        idx_int = int(re.findall(idx_regex, p)[0])
        return idx_int
    else:
        idx_dict = {re.findall(kind_regex, p)[0]: int(re.findall(idx_regex, p)[0])
                    for p in patterns}
        return idx_dict


def gen_derived_fnames(base_fnames,
                       name='log',
                       extension='json',
                       dname=None,
                       indexed=True,
                       sep='_'):
    """
    Generate derived filenames

    :return:
    """
    if not isinstance(base_fnames, list):
        base_fnames = [base_fnames]

    if isinstance(base_fnames[0], tuple):
        # We have a list of tuples, assumed (index, fname)
        base_fnames = [t[1] for t in base_fnames]

    fnames = []
    for fname in base_fnames:
        fname = insert_msg_in_fname(fname, name, sep=sep)

        if extension is not None:
            fname = alter_extension_fname(fname, extension)

        if dname is not None:
            fname = alter_directory_fname(fname, dname)

        fnames.append(fname)

    if indexed:
        fnames = [(extract_idx_from_string(f), f) for f in fnames]

    return fnames


def gen_output_data_fnames(test_data_fnames,
                           output_data_dir,
                           nb_queries=1,
                           sep="_"):
    """
    Generate filenames for the outputdata.

    These filenames depend on the number of queries and the number of folds.
    """

    q_appendix = [gen_appendix(i, kind='query')
                  for i in range(nb_queries)]

    result = []
    for fname in test_data_fnames:
        [base, ext] = os.path.splitext(os.path.basename(fname))

        for q_idx in range(nb_queries):
            res_string = base + sep + q_appendix[q_idx] + ext
            res = os.path.join(output_data_dir, res_string)
            result.append(res)

    return result


def collect_fnames_from_folder(folder,
                               criteria=None,
                               disjoint=False,
                               indexed=False):
    fnames = os.listdir(folder)

    if criteria is None:
        criteria = ['']
    elif isinstance(criteria, str):
        criteria = [criteria]
    else:
        pass
    assert isinstance(criteria, list)

    if disjoint:
        # Criterion one OR criterion two in fname
        res = []
        for criterion in criteria:
            res.extend([f for f in fnames if criterion in f])
        fnames = res
    else:
        # Criterion one AND criterion two in fname
        for criterion in criteria:
            fnames = [f for f in fnames if criterion in f]

    fnames = [os.path.join(folder, f) for f in fnames]
    fnames.sort()

    if indexed:
        fnames = [(extract_idx_from_string(f),f) for f in fnames]

    return fnames


# New methods TODO: Some of the above methods can be simplified using these!
def insert_msg_in_fname(fname, msg, sep='_'):
    [name, ext] = os.path.splitext(fname)

    if isinstance(msg, (list, tuple, set)):
        for m in msg:
            fname = insert_msg_in_fname(fname, m, sep=sep)
    else:
        assert isinstance(msg, str)
        fname = name + sep + msg + ext

    return fname


def alter_extension_fname(fname, extension):
    [name, _] = os.path.splitext(fname)

    if extension.startswith('.') :
        fname = name + extension
    elif extension == '':
        fname = name
    else:
        fname = name + '.' + extension
    return fname


def alter_directory_fname(fname, dname):
    base = os.path.basename(fname)
    new_fname = os.path.join(dname, base)
    return new_fname


# Detect idxs
def detect_largest_idx_in_directory(dname):
    """
    Detect largest idx in directory

    Useful to detect the last experiment conducted.

    This assessment is based on which folder is present.

    Parameters
    ----------
    dname: str
        Path of the directory which has to examined

    Returns
    -------

    """
    # TODO: Robust version relying on regex!

    idx = 0
    idx_in_dir = [os.path.splitext(x)[0] for x in os.listdir(dname)]
    idx_in_dir.sort(reverse=True)

    for id_folder in idx_in_dir:
        if id_folder.isdigit():
            idx = int(id_folder)
            break

    return idx
