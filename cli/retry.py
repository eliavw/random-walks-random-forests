#!/usr/bin/env python
"""
Entry point.

Command line script that;
    1. Makes a RunExp
    2. Extracts its commands
    3. Runs those commands through run_remote.
"""

# region Preliminaries
# Imports
import argparse
import json
import subprocess
import os
import pandas as pd
import pickle as pkl
import sys

from os.path import dirname, abspath, relpath

# Custom imports
cli_dir = dirname(abspath(__file__))
root_dir = dirname(cli_dir)
for e in {'libs', 'src'}:
    sys.path.append(os.path.join(root_dir, e))

from exp.runner.RunExp import RunExp
from exp.utils.extra import debug_print # For debugging purposes
# endregion

VERBOSITY = 1


def main(commands_fname, local=True):

    # Actions
    commands_df = pd.read_csv(commands_fname)
    env = _build_env(commands_df)

    # Execution
    _execute_command_subprocess(local=local, **env)

    return


# Helpers
def _build_env(commands_df):
    """
    Environment vars to go to bach script.

    Parameters
    ----------
    commands_df

    Returns
    -------

    """

    # Init
    env = {}

    # Define (hardcoding)
    env['commands_fname'] = os.path.join(cli_dir, 'commands.csv')
    env['nodefile_fname'] = os.path.join(cli_dir, 'nodefile')
    env['exe_atom_fname'] = os.path.join(root_dir, 'cmd', 'execute_atom.py')
    env['run_remote_fname'] = os.path.join(root_dir, 'cmd', 'run_remote')
    env['run_local_fname'] = os.path.join(root_dir, 'cmd', 'run_local')
    env['remote_conda'] = "/cw/dtaijupiter/NoCsBack/dtai/elia/00_Software/anaconda/bin"
    env['local_conda'] = "/home/elia/Software/anaconda3/bin"

    # Derive
    env['nb_procs'] = str(commands_df.shape[0]-1)

    return env


def _execute_command_subprocess(local=True, **kwargs):

    env = {**kwargs, **os.environ.copy()}

    # Remote or Local
    if local:
        bash = env['run_local_fname']
    else:
        bash = env['run_remote_fname']

    # Execute
    subprocess.call(bash, env=env)

    return


# For executable script
if __name__ == '__main__':

    # Extracting options
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands', '-c',
                        help='commands_fname_outer_scope')
    parser.add_argument('--local', '-l',
                        help='local_outer_scope, local yes/no',
                        action="store_true")

    args = parser.parse_args()

    commands_fname_outer_scope = args.commands
    local_outer_scope = args.local

    msg = """
    We are running local: {}
    """.format(local_outer_scope)
    debug_print(msg, V=VERBOSITY)

    assert isinstance(commands_fname_outer_scope, str)

    main(commands_fname_outer_scope, local=local_outer_scope)
