import os
import pandas as pd
import pickle as pkl
import warnings

from os.path import splitext

from .Runner import Runner
from ..eval.collect import aggregate_outputs
from ..utils.config import filter_kwargs_function
from ..utils.extra import (debug_print,
                           generate_keychain)
from ..utils.filesystem import (ensure_dir,
                                gen_derived_fnames,
                                make_fname)

VERBOSITY = 0


class RunExp(Runner):

    def __init__(self, **kwargs):
        super(RunExp, self).__init__(script='run_exp',
                                     **kwargs)

        # What this produces
        self.prods = {'outputs', 'config', 'logs', 'results', 'runner'}

        # Outputs that afterwards can be aggregated
        self.aggr_outputs = {'results', 'timings', 'mod_config', 'qry_codes'}
        return

    @classmethod
    def load(cls, **kwargs):

        # Preliminaries
        helper = cls()
        helper.make_config(**kwargs)

        kind = helper.name
        runner_fname = helper.get_fname(kind)

        # Actually load the experiment
        if os.path.isfile(runner_fname):
            with open(runner_fname, 'rb') as f:
                runner = pkl.load(f)
            del helper
        else:
            msg = """
            Could not load the requested instance: {}
            """.format(runner_fname)
            raise IOError(msg)
        return runner

    # Config - core
    def default_config(self, **kwargs):
        """
        Default configuration for our script.

        Typically, this concerns I/O stuff. `Default status' is of course
        a subjective assessment.

        :return:
        """
        # Basic defaults
        self.config['timeout'] = kwargs.get('timeout', 60)
        self.config['child'] = kwargs.get('child', 'RunMercs')

        # IO defaults
        io_kwargs = filter_kwargs_function(self.default_io_config, **kwargs)
        self.default_io_config(**io_kwargs)

        return

    def update_config(self, **kwargs):
        """
        Update the config dict.

        There are essentially three modes:
            1. Configs to pass on. We do not unpack those values and lock
            them into one entry in the dict.
            2. Configs that are directly relevant to the the current object,
            those we need to unpack and assign correctly.
            3. Special cases. This usually means configurations that are a bit
            more complicated to generate. I.e., actual work that needs to be
            done. That work is outsourced to dedicated methods, and those methods
            simply return parameters that then go in the standard workflow.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        child = self.config['child']

        # Configs that we just pass on (i.e., do not unpack these!)
        for kind in {child, 'explore'}:
            super(RunExp, self).update_config(kind=kind,
                                              hierarchical=False,
                                              overwrite=False,
                                              **kwargs)
            kwargs = {k: v for k, v in kwargs.items()
                      if kind not in k}  # Exclude kwargs that you pass on this way

        # Update config Exp
        super(RunExp, self).update_config(hierarchical=True,
                                          **kwargs)

        return

    def save_config(self, overwrite=True):
        """
        Save the configuration file.

        The location where it needs to be saved is indicated in the config
        file itself.

        What is special is the fact that the products might have to be erased.

        Returns
        -------

        """

        # TODO: Take care about what you erase here
        for d in self.prods:
            key = generate_keychain(['prod', d])
            dname = self.config['io']['dirs'][key]
            if overwrite: ensure_dir(dname)

        return super(RunExp, self).save_config()

    # Config - defaults
    def default_io_config(self, root_dir=None, dataset=None, idx=None):
        """
        Generate default io config dictionary

        Parameters
        ----------
        root_dir: str
            Path to the root dir of this experiment
        dataset: str
            Name of the dataset with which this particular experiment is run
        idx: int
            Index of the current experiment

        Returns
        -------

        """
        self.config['io'] = {}

        # io-dirs
        d = super(RunExp, self).default_io_config_dirs(root_dir=root_dir,
                                                       dataset=dataset,
                                                       idx=idx)
        self.config['io']['dirs'] = d

        # io-files
        d = self._default_io_config_file()
        self.config['io']['file'] = d

        return

    # Config - helpers
    def _default_io_config_file(self, dirs=None):

        # Extract config
        script = self.script
        name = self.name
        aggr_outputs = self.aggr_outputs

        # Initialize
        d = {}
        if dirs is None:
            dirs = self.config['io']['dirs']

        # Commands (cmd)
        d['script'] = make_fname(name=script,
                                 extension='py',
                                 dname=dirs['cmd'])

        # Products (prod)
        for n in {'commands', 'outp_fns'}:
            d[n] = make_fname(name=n,
                              extension='csv',
                              dname=dirs['prod-outputs'])

        for n in aggr_outputs:
            d[n] = make_fname(name=n,
                              extension='pkl',
                              dname=dirs['prod-results'])

        for n in {'config', 'exploration'}:
            d[n] = make_fname(name=n, dname=dirs['prod-config'])

        d['log'] = gen_derived_fnames(d['script'],
                                      name='log',
                                      extension='',
                                      dname=dirs['prod-logs'],
                                      indexed=False)
        assert len(d['log']) == 1
        d['log'] = d['log'][0]

        d[name] = make_fname(name=name,
                             extension='pkl',
                             dname=dirs['prod-runner'])

        return d

    # Run
    def generate_commands(self, **kwargs):

        # Preliminaries
        self.update_config(**kwargs)
        self.save_config(overwrite=True)

        # Extract config
        script_fname = self.config['io']['file']['script']
        config_fname = self.config['io']['file']['config']
        log_fname = self.config['io']['file']['log']

        timeout = self.config['timeout']

        # Init list of tuples
        head_tuple = ('script_fname',
                      'config_fname',
                      'log_fname',
                      'timeout')
        data_tuple = ('script_fname',
                      'config_fname',
                      'log_fname',
                      60)
        tuple_list = [data_tuple]

        # Build list of tuples
        tuple_list[0] = (script_fname,
                         config_fname,
                         log_fname,
                         timeout)

        return head_tuple, tuple_list

    # Commands
    def load_output(self, kind='commands', new_root=None):
        output_fname = self.config['io']['file'][kind]

        _, ext = splitext(output_fname)

        if new_root is not None:
            old_root = self.config['io']['dirs']['root']
            output_fname = output_fname.replace(old_root, new_root)

        if ext in {'.csv'}:
            df = pd.read_csv(output_fname, index_col=0)
        elif ext in {'.pkl'}:
            df = pd.read_pickle(output_fname)
        else:
            msg = """
            Did not recognize the extension of this output file: {}
            """.format(output_fname)
            raise IOError(msg)

        return df

    def save_output(self, kind='commands'):
        output_fname = self.config['io']['file'][kind]
        self.__getattribute__(kind).to_pickle(output_fname)
        return

    def aggregate_outputs(self, save=False):
        df_fns = self.load_output(kind='outp_fns')  # Load DF that holds fnames of outputs
        aggr_outputs = self.aggr_outputs

        for name in aggr_outputs:
            output_df = aggregate_outputs(df_fns, kind=name)
            self.__setattr__(name, output_df)

            if save:
                self.save_output(kind=name)
        return
