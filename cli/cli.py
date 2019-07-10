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
for e in {"libs", "src"}:
    sys.path.append(os.path.join(root_dir, e))

from exp.runner.RunExp import RunExp
from exp.utils.extra import debug_print  # For debugging purposes

# endregion

VERBOSITY = 0

# Change with appropriate python paths
LOCAL_CONDA = "/home/elia/miniconda/envs/rwrf/bin"
REMOTE_CONDA = "/cw/dtaijupiter/NoCsBack/dtai/elia/00_Software/anaconda/bin"


def main(param_file_or_folder, local=True):
    # Inputs
    param_fnames = _build_param_fnames(param_file_or_folder)

    # Actions
    commands_df = _build_commands(param_fnames)
    env = _build_env(commands_df)

    # Outputs
    commands_df.to_csv(env["commands_fname"])

    # Execution
    _execute_command_subprocess(local=local, **env)

    return


# Helpers
def _build_param_fnames(param_file_or_folder, try_config=True):
    """
    Extract names of parameter json files.

    Parameters
    ----------
    param_file_or_folder: str
        Parameter .json file or folder containing multiple such files

    Returns
    -------

    """

    if os.path.isfile(param_file_or_folder):
        param_fnames = [param_file_or_folder]
    elif os.path.isdir(param_file_or_folder):
        param_fnames = os.listdir(param_file_or_folder)
        param_fnames = [os.path.join(param_file_or_folder, f) for f in param_fnames]
    elif try_config:
        # Check the config folder.
        param_file_or_folder_new = os.path.join("config", param_file_or_folder)
        param_fnames = _build_param_fnames(param_file_or_folder_new, try_config=False)
    else:
        msg = """
        Input parameter is neither file nor folder: {}
        """.format(
            param_file_or_folder
        )
        raise ValueError(msg)

    return param_fnames


def _build_commands(fnames, shuffle=True):
    """
    Build commands from parameter .json file(s)

    Commands are generated by an instance exp of the Exp() class. This happens
    for each .json file. Afterwards, the commands are extracted from exp, and
    collected in a big list.

    Parameters
    ----------
    fnames

    Returns
    -------

    """

    commands_df = pd.DataFrame()
    for fname in fnames:
        msg = """
        Looking at parameters file: {}
        """.format(
            fname
        )
        debug_print(msg, V=VERBOSITY)

        with open(fname, "r") as f:
            parameters = json.load(f)

        # Init
        exp = RunExp()

        # Config
        exp.make_config(**parameters)
        exp.save_config()

        # Generate commands of RunExp's children
        exp.run()

        # Load commands
        commands_df = commands_df.append(exp.load_output(kind="commands"))

        exp_fname = exp.config["io"]["file"]["RunExp"]
        with open(exp_fname, "wb") as f:
            pkl.dump(exp, f)

        del exp

    if shuffle:
        # Shuffle all commands, cf. https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        commands_df = commands_df.sample(frac=1).reset_index(drop=True)

    return commands_df


def _build_env(commands_df):

    # Init
    env = {}

    # Define (hardcoding)
    env["commands_fname"] = os.path.join(cli_dir, "commands.csv")
    env["nodefile_fname"] = os.path.join(cli_dir, "nodefile")
    env["exe_atom_fname"] = os.path.join(root_dir, "cmd", "execute_atom.py")
    env["run_remote_fname"] = os.path.join(root_dir, "cmd", "run_remote")
    env["run_local_fname"] = os.path.join(root_dir, "cmd", "run_local")
    env["remote_conda"] = REMOTE_CONDA
    env["local_conda"] = LOCAL_CONDA

    # Derive
    env["nb_procs"] = str(commands_df.shape[0] - 1)

    return env


def _execute_command_subprocess(local=True, **kwargs):

    env = {**kwargs, **os.environ.copy()}

    # Remote or Local
    if local:
        bash = env["run_local_fname"]
    else:
        bash = env["run_remote_fname"]

    # Execute
    subprocess.call(bash, env=env)

    return


def cli_parser():
    """
    CLI Parser.

    Returns
    -------

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="config_file_or_folder_outer_scope")
    parser.add_argument(
        "--local", "-l", help="local_outer_scope, local yes/no", action="store_true"
    )

    return parser


# For executable script
if __name__ == "__main__":

    parser = cli_parser()
    args = parser.parse_args()

    config_file_or_folder_outer_scope = args.config
    local_outer_scope = args.local

    assert isinstance(config_file_or_folder_outer_scope, str)

    msg = """
    We are running local: {}
    """.format(
        local_outer_scope
    )
    debug_print(msg, V=VERBOSITY)

    main(config_file_or_folder_outer_scope, local=local_outer_scope)