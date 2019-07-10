import numpy as np

from ..experimenter.monitor import Logfile, TimeLimit
from ..experimenter.process import Process

from .utils import debug_print
VERBOSITY = 0


# Main
def generate_monitor(log_fname, timeout):
    """
    Create a monitor to go with a command.

    A monitor's job is to monitor a process. So, instead of just executing a
    command (i.e., running a process), we first build a monitor, and then pass
    monitor and command together to a dedicated Process class.

    This Process class executes the command, but also uses the monitor to
    capture logs and guard the timeout.

    Parameters
    ----------
    log_fname: str
        Filename of log life
    timeout: int
        Timeout in seconds. Script is automatically aborted after this period.

    Returns
    -------

    """
    assert isinstance(log_fname, str)
    assert isinstance(timeout, (int, np.int64))

    msg = """
    File is:        {}\n
    log_fname is:   {}\n
    timeout is:     {}\n
    """.format(__file__, log_fname, timeout)
    debug_print(msg, V=VERBOSITY)

    monitors = [Logfile(log_fname),
                TimeLimit(timeout)]

    return monitors


def run_process(command, monitors=None, cwd=None):
    """
    Execute the command, monitored by the specified monitors.

    Parameters
    ----------
    command: list, shape(nb_strings, )
        List of strings that constitute the command to be entered in the
        terminal
    monitors: list, shape (nb_monitors)
        List of monitors, which will be used to monitor the process that
        executes the command.

    Returns
    -------

    """
    if monitors is None:
        msg = """
        Running process without monitors
        """
        print(msg)
    p = Process(command, monitors=monitors, cwd=cwd)  # Init Process
    return p.run()
