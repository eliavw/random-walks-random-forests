# region Preliminaries
# Imports
import os
import pandas as pd
import sys

from os.path import dirname, abspath, relpath
from inspect import signature

# Custom imports
cmd_dir = dirname(abspath(__file__))
root_dir = dirname(cmd_dir)
for e in {'libs', 'src'}:
    sys.path.append(os.path.join(root_dir, e))

from exp.utils.run import run_script
from exp.utils.extra import debug_print # For debugging purposes
# endregion

VERBOSITY = 0


def main(csv_fname, cmd_idx):
    """
    Run single command from csv file that specifies many commands.

    The command that should be run corresponds to a row in the .csv file
    with all commands. The specific command that should be run is indicated
    by the row idx.

    Parameters
    ----------
    csv_fname: str
        Filename of the csv containing all commands
    cmd_idx: int
        Index of row that corresponds to command to be run.

    Returns
    -------

    """
    assert isinstance(cmd_idx, int)
    assert isinstance(csv_fname, str)

    # Extract command
    df = pd.read_csv(csv_fname, index_col=0)
    head_tuple = tuple(df.columns)
    data_tuple = tuple(df.iloc[cmd_idx])
    param_dict = {k: v for k, v in zip(head_tuple, data_tuple)}

    msg = """
    param_dict: {}
    """.format(param_dict)
    debug_print(msg,V=VERBOSITY)

    # Run command
    sig = signature(run_script)
    ba = sig.bind(**param_dict)

    run_script(*ba.args, **ba.kwargs)
    return


if __name__ == '__main__':
    # Extract parameters
    csv_fname_outer_scope = sys.argv[1]
    cmd_idx_outer_scope = int(sys.argv[2])

    # Run main
    main(csv_fname_outer_scope, cmd_idx_outer_scope)
