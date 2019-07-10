"""
Run exp script.

Explore a part of the parameter space.

For each parameter tuple, generate
a command that would execute exactly that experiment.

Or, alternatively, run that experiment immediately
"""

# region Preliminaries
import itertools as itools
import json
import os
import sys
import pandas as pd

from os.path import dirname

# Custom imports
root_dir = dirname(dirname(__file__))
for e in {"libs", "src"}:
    sys.path.append(os.path.join(root_dir, e))

from exp.runner.RunMercs import RunMercs
from exp.runner.RunPxS import RunPxS
from exp.runner.RunPxL import RunPxL
from exp.runner.RunPxLParallel import RunPxLParallel
from exp.runner.RunPxSParallel import RunPxSParallel
from exp.runner.MakeModel import MakeModel
from exp.runner.RunVersaDummy import RunVersaDummy
from exp.runner.RunModulo import RunModulo
from exp.utils.filesystem import detect_largest_idx_in_directory
from exp.utils.extra import debug_print  # For debugging purposes

# endregion

VERBOSITY = 0
PRECISION = 6


def main(config_fname):
    # Load config
    config = load_config(config_fname)
    child = config["child"]

    io_config = config["io"]

    msg = """
    Directories in this experiment are: {}
    """.format(
        config["io"]["dirs"]
    )
    debug_print(msg, level=2, V=VERBOSITY)

    # Make exploration
    explore_config = config["explore"]
    start_idx = config.get("start-idx", None)
    start_idx = determine_start_idx(start_idx, io_config)

    explore_config = explore(explore_config, start_idx)

    # Generate commands
    static_child_config = config[child]  # Static parameters
    cmd_ht, cmd_tl, ofn_ht, ofn_tl = generate_outputs(
        child, explore_config, static_child_config
    )

    # Save outputs
    save_explore(explore_config, io_config)
    save_outputs(cmd_ht, cmd_tl, io_config, kind="commands")
    save_outputs(ofn_ht, ofn_tl, io_config, kind="outp_fns")

    return


# Inputs
def load_config(config_fname):
    with open(config_fname, "r") as f:
        config = json.load(f)
    return config


# Actions
def determine_start_idx(start_idx, io_config):
    assert isinstance(start_idx, (int, type(None)))

    if start_idx is None:
        prod_dir = dirname(dirname(io_config["dirs"]["prod"]))

        msg = """
        io_config['dirs']['prod']: {}
        prod_dir assumed to be: {}
        """.format(
            io_config["dirs"]["prod"], prod_dir
        )
        debug_print(msg, V=VERBOSITY)

        subdirs = [os.path.join(prod_dir, d) for d in os.listdir(prod_dir)]
        start_idx = [detect_largest_idx_in_directory(d) for d in subdirs]
        start_idx = max(start_idx) + 1

        msg = """
        Automatically detected start-idx: {}
        """.format(
            start_idx
        )
        debug_print(msg, V=VERBOSITY)

    return start_idx


def explore(explore_config, start_idx):
    explore_config = _explore_ranges_to_grid(explore_config, start_idx)
    assert _assert_explore_grid(explore_config)

    return explore_config


def _explore_ranges_to_grid(d, start_idx):
    """
    Convert ranges to a grid

    Convert a dictionary which specifies ranges for its parameters into
    a dictionary that contains a grid of parameters.

    Parameters
    ----------
    d: dict
        Dictionary where each key corresponds to a parameter. Each value
        corresponds to the range of values that should be explored for that
        parameter.
        E.g.:
            {'height':  [1.80, 1.90, 2.00],
             'weight':  [70.0, 90.0]}

    Returns
    -------
    d: dict
        Dictionary where each key corresponds to a parameter. Each value
        corresponds to a list of values for that parameter. The i-th element
        in that list is the value of that parameter corresponding to the i-th
        point to be explored in the parameter grid.
        E.g.:
            {'height':  [1.80, 1.80, 1.90, 1.90, 2.00, 2.00],
             'weight':  [70.0, 90.0, 70.0, 90.0, 70.0, 90.0]}

    """

    assert isinstance(d, dict)
    assert isinstance(start_idx, int)

    # Wrap single values into a list
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}

    # Initialize keys and iterator lists
    keys_list, iter_list = [], []
    for k, param_v in d.items():
        keys_list.append(k)
        iter_list.append(iter(param_v))

    # Cartesian product of all iterators
    cartesian_iterator = itools.product(*iter_list)

    # Build final dict
    explore_grid = {k: [] for k in keys_list}
    explore_grid["idx"] = []
    for point_idx, cartesian_tuple in enumerate(cartesian_iterator):
        idx = start_idx + point_idx
        explore_grid["idx"].append(idx)

        for param_idx, param_v in enumerate(cartesian_tuple):
            key = keys_list[param_idx]
            explore_grid[key].append(param_v)

    return explore_grid


def _assert_explore_grid(explore_grid):
    """
    Assert correctness of exploration grid of parameters.

    Parameters
    ----------
    explore_grid: dict
        Dictionary where each key corresponds to a parameter. Each value
        corresponds to a list of values for that parameter. The i-th element
        in that list is the value of that parameter corresponding to the i-th
        point to be explored in the parameter grid.
        E.g.:
            {'height':  [1.80, 1.80, 1.90, 1.90, 2.00, 2.00],
             'weight':  [70.0, 90.0, 70.0, 90.0, 70.0, 90.0]}

    Returns
    -------

    """

    nb_exps = set([len(v) for k, v in explore_grid.items()])
    return len(nb_exps) == 1


def generate_outputs(child, explore_config, static_child_config):
    # Prelims
    nb_exps = len(explore_config["idx"])
    assert nb_exps > 0

    # Cast dict to correct list
    all_child_config = [
        {k: v[idx] for k, v in explore_config.items()} for idx in range(nb_exps)
    ]

    all_child_config = [{**x, **static_child_config} for x in all_child_config]

    # First child
    runner = _runner_init(child, all_child_config[0])
    cmd_head_tuple, cmd_tuple_list = runner.generate_commands()
    ofn_head_tuple, ofn_tuple_list = runner.generate_outp_fns()
    del runner

    # Other children
    for child_config in all_child_config[1:]:
        runner = _runner_init(child, child_config)
        _, x = runner.generate_commands()
        cmd_tuple_list.extend(x)

        _, y = runner.generate_outp_fns()
        ofn_tuple_list.extend(y)
        del runner

    return cmd_head_tuple, cmd_tuple_list, ofn_head_tuple, ofn_tuple_list


def _runner_init(child, child_config):
    if child in {"RunMercs"}:
        runner = RunMercs()
    elif child in {"RunSmile"}:
        runner = RunPxS()
    elif child in {"RunPxL"}:
        runner = RunPxL()
    elif child in {"RunPxLParallel"}:
        runner = RunPxLParallel()
    elif child in {"RunPxSParallel"}:
        runner = RunPxSParallel()
    elif child in {"MakeModel"}:
        runner = MakeModel()
    elif child in {"RunVersaDummy"}:
        runner = RunVersaDummy()
    elif child in {"RunModulo"}:
        runner = RunModulo()
    else:
        msg = """
        Did not recognize child: {}
        We cannot build such an entity, let alone use it to generate
        commands.
        """.format(
            child
        )
        raise ValueError(msg)

    # Config
    runner.make_config(**child_config)
    runner.save_config()
    return runner


# Outputs
def save_explore(explore_config, io_config):
    with open(io_config["file"]["exploration"], "w") as f:
        json.dump(explore_config, f, indent=4)

    return


def save_outputs(head_tuple, tuple_list, io_config, kind="commands"):
    fname = io_config["file"][kind]

    msg = """
    Head Tuple:             {}
    len(tuple_list[0]):     {}
    tuple_list[0]:          {}
    """.format(
        head_tuple, len(tuple_list[0]), tuple_list[0]
    )
    debug_print(msg, V=VERBOSITY)

    df = pd.DataFrame.from_records(tuple_list, columns=head_tuple)
    df.to_csv(fname)
    return


# For executable script
if __name__ == "__main__":
    config_fname_outer_scope = sys.argv[1]
    main(config_fname_outer_scope)
