import pandas as pd
import warnings

from .RunPxS import RunPxS
from ..utils.config import filter_kwargs_function
from ..utils.extra import (debug_print,
                           generate_keychain)
from ..utils.filesystem import (collect_fnames_from_folder,
                                ensure_dir,
                                gen_derived_fnames,
                                make_fname,
                                gen_appendix,
                                insert_msg_in_fname)

VERBOSITY = 0


class RunPxSParallel(RunPxS):

    def __init__(self, **kwargs):
        super(RunPxSParallel, self).__init__(**kwargs)
        self.script = 'run_PxS_parallel'
        return

    # Config
    def default_config(self, **kwargs):
        """
        Default configuration for our script.

        Typically, this concerns I/O stuff.
        `Default` status of course remains a subjective assessment.

        Parameters
        ----------
        kwargs: dict
            Parameters provided by the user.

        Returns
        -------

        """
        self.config['nb_queries'] = kwargs.get('nb_queries', 10)
        super(RunPxSParallel, self).default_config(**kwargs)
        return

    # Run
    def generate_commands(self, **kwargs):

        # Preliminaries
        self.update_config(**kwargs)
        self.save_config(overwrite=True)

        # Extract config
        script_fname = self.config['io']['file']['script']
        config_fname = self.config['io']['file']['config']
        log_fnames = self.config['io']['file']['logs']

        folds = self.config['folds']
        timeout = self.config['timeout']
        nb_queries = self.config['nb_queries']

        head_tuple = ('script_fname',
                      'config_fname',
                      'log_fname',
                      'q_idx',
                      'fold',
                      'timeout')
        data_tuple = ('script_fname',
                      'config_fname',
                      'log_fname',
                      0,
                      0,
                      60)
        tuple_list = [data_tuple] * (len(folds)*nb_queries)
        for f_idx, fold in enumerate(folds):
            for q_idx in range(nb_queries):
                log_fname = [t[2] for t in log_fnames
                             if t[0] == fold
                             if t[1] == q_idx][0]

                param_tuple = (script_fname,
                               config_fname,
                               log_fname,
                               q_idx,
                               fold,
                               timeout)
                cmd_idx = (f_idx * nb_queries) + q_idx
                tuple_list[cmd_idx] = param_tuple

        return head_tuple, tuple_list

    def generate_outp_fns(self):
        idx = self.config['idx']
        folds = self.config['folds']
        nb_queries = self.config['nb_queries']

        mod_config_fname = self.get_fname('config')
        qry_codes_fname = self.get_fname('qry-codes')

        results_fnames = self.get_fname('results')
        timings_fnames = self.get_fname('timings')

        head_tuple = ('idx',
                      'f_idx',
                      'q_idx',
                      'results_fname',
                      'timings_fname',
                      'mod_config_fname',
                      'qry_codes_fname')
        data_tuple = (0,
                      0,
                      0,
                      'str',
                      'str',
                      'str',
                      'str')
        tuple_list = [data_tuple] * (len(folds)*nb_queries)

        for f_idx, fold in enumerate(folds):
            for q_idx in range(nb_queries):
                results_fname = [t[2] for t in results_fnames
                                 if t[0] == fold
                                 if t[1] == q_idx][0]
                timings_fname = [t[2] for t in timings_fnames
                                 if t[0] == fold
                                 if t[1] == q_idx][0]

                outp_tuple = (idx,
                              f_idx,
                              q_idx,
                              results_fname,
                              timings_fname,
                              mod_config_fname,
                              qry_codes_fname)

                output_idx = (f_idx * nb_queries) + q_idx
                tuple_list[output_idx] = outp_tuple

        return head_tuple, tuple_list

    # Config helpers
    def _default_io_config_file(self, dirs=None):
        d = super(RunPxSParallel, self)._default_io_config_file(dirs=dirs)
        nb_queries = self.config['nb_queries']

        for k in {'logs', 'timings', 'results'}:
            d[k] = self._adapt_tuple_list(d[k], nb_queries)

        return d

    @staticmethod
    def _adapt_tuple(tup, q_idx):
        q_appendix = gen_appendix(q_idx, kind='Q')
        new_fname = insert_msg_in_fname(tup[1], q_appendix, sep='_')

        new_tuple = (tup[0], q_idx, new_fname)
        return new_tuple

    def _adapt_tuple_list(self, tuple_list, nb_queries):

        new_tuple_list = []

        for tup in tuple_list:
            for q_idx in range(nb_queries):
                new_tuple = self._adapt_tuple(tup, q_idx)
                new_tuple_list.append(new_tuple)

        return new_tuple_list
