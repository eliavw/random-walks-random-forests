from .Runner import Runner
from ..utils.config import filter_kwargs_function
from ..utils.extra import (debug_print,
                           generate_keychain)
from ..utils.filesystem import (collect_fnames_from_folder,
                                ensure_dir,
                                gen_derived_fnames,
                                make_fname)

VERBOSITY = 0


class RunVersaDummy(Runner):

    def __init__(self, **kwargs):

        init_kwargs = filter_kwargs_function(super(RunVersaDummy, self).__init__,
                                             **kwargs)
        super(RunVersaDummy, self).__init__(script='run_VersaDummy', **init_kwargs)

        self.prods = {'config', 'logs', 'results', 'timings'}  # What this produces

        return

    # Config
    def default_config(self, **kwargs):
        """
        Default configuration for our script.

        Typically, this concerns I/O stuff. Default status is of course
        a subjective assessment.

        Parameters
        ----------
        kwargs: dict
            Parameters provided by the user.

        Returns
        -------

        """

        # Basic defaults
        self.config['child'] = kwargs.get('child', 'VersaDummy')
        self.config['folds'] = kwargs.get('folds', None)
        self.config['timeout'] = kwargs.get('timeout', 60)

        # IO defaults
        io_kwargs = filter_kwargs_function(self.default_io_config, **kwargs)
        self.default_io_config(**io_kwargs)

        # Query defaults
        self.default_qry_config()

        # Eval defaults
        self.default_eval_config()

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
        for kind in {child, 'eval'}:
            super(RunVersaDummy, self).update_config(kind=kind,
                                                     hierarchical=False,
                                                     overwrite=False,
                                                     **kwargs)
            kwargs = {k: v for k, v in kwargs.items()
                      if kind not in k}  # Exclude kwargs passed on this way

        # Update config RunVersaDummy
        super(RunVersaDummy, self).update_config(hierarchical=True,
                                                 **kwargs)

        # Special cases - query (qry)
        qry_kwargs = self._generate_qry_params()
        super(RunVersaDummy, self).update_config(hierarchical=True,
                                                 **qry_kwargs)

        # Special cases - model (mod)
        mod_kwargs = self._generate_mod_params()
        super(RunVersaDummy, self).update_config(hierarchical=True,
                                                 **mod_kwargs)

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

        super(RunVersaDummy, self).save_config()
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

        head_tuple = ('script_fname',
                      'config_fname',
                      'log_fname',
                      'fold',
                      'timeout')
        data_tuple = ('script_fname',
                      'config_fname',
                      'log_fname',
                      0,
                      60)
        tuple_list = [data_tuple] * len(folds)
        for f_idx, fold in enumerate(folds):
            log_fname = [t[1] for t in log_fnames if t[0] == fold][0]

            param_tuple = (script_fname,
                           config_fname,
                           log_fname,
                           fold,
                           timeout)
            tuple_list[f_idx] = param_tuple

        return head_tuple, tuple_list

    def generate_outp_fns(self):
        idx = self.config['idx']
        folds = self.config['folds']

        mod_config_fname = self.get_fname('config')
        qry_codes_fname = self.get_fname('qry-codes')

        results_fnames = self.get_fname('results')
        timings_fnames = self.get_fname('timings')

        head_tuple = ('idx',
                      'f_idx',
                      'results_fname',
                      'timings_fname',
                      'mod_config_fname',
                      'qry_codes_fname')

        data_tuple = (0,
                      0,
                      'str',
                      'str',
                      'str',
                      'str')
        tuple_list = [data_tuple] * len(folds)

        for f_idx, fold in enumerate(folds):
            results_fname = [t[1] for t in results_fnames if t[0] == fold][0]
            timings_fname = [t[1] for t in timings_fnames if t[0] == fold][0]

            outp_tuple = (idx,
                          f_idx,
                          results_fname,
                          timings_fname,
                          mod_config_fname,
                          qry_codes_fname)

            tuple_list[f_idx] = outp_tuple

        return head_tuple, tuple_list

    # Default configs
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
        d = super(RunVersaDummy, self).default_io_config_dirs(root_dir=root_dir,
                                                              dataset=dataset,
                                                              idx=idx)
        self.config['io']['dirs'] = d

        # io-files
        d = self._default_io_config_file()
        self.config['io']['file'] = d

        return

    def default_qry_config(self):
        self.config['qry'] = {}
        return

    def default_eval_config(self):
        self.config['eval'] = {}
        return

    # Config helpers
    def _default_io_config_file(self, dirs=None):

        # Initialize
        d = {}
        if dirs is None:
            dirs = self.config['io']['dirs']

        folds = self.config['folds']
        child = self.config['child']

        # Commands (cmd)
        d['script'] = make_fname(name=self.script,
                                 extension='py',
                                 dname=dirs['cmd'])

        # Resources (resc)
        d['train_data'] = collect_fnames_from_folder(dirs['resc-data-ds'],
                                                     criteria=['Train', 'bayesfusion'],
                                                     indexed=True)
        d['test_data'] = collect_fnames_from_folder(dirs['resc-data-ds'],
                                                    criteria=['Test', 'bayesfusion'],
                                                    indexed=True)

        # Things related to folds
        if folds is not None:
            d['train_data'] = [t for t in d['train_data'] if t[0] in folds]
            d['test_data'] = [t for t in d['test_data'] if t[0] in folds]

        if not (len(d['train_data']) == len(d['test_data']) > 0):
            msg = """
            Fatal inconsistency encountered when loading train and test data.
            Train data: {}
            Test data: {}
            folds: {}
            """.format(d['train_data'], d['test_data'], folds)
            raise ValueError(msg)

        # Products (prod)
        d['config'] = make_fname(name=self.name, dname=dirs['prod-config'])

        d['mod-config'] = gen_derived_fnames(d['train_data'],
                                             name=child,
                                             extension='pkl',
                                             dname=dirs['prod-config'],
                                             indexed=True)

        d['logs'] = gen_derived_fnames(d['test_data'],
                                       name='log',
                                       extension='',
                                       dname=dirs['prod-logs'],
                                       indexed=True)

        d['timings'] = gen_derived_fnames(d['test_data'],
                                          name='times',
                                          extension='csv',
                                          dname=dirs['prod-timings'],
                                          indexed=True)

        d['results'] = gen_derived_fnames(d['test_data'],
                                          name='results',
                                          extension='csv',
                                          dname=dirs['prod-results'],
                                          indexed=True)

        return d

    def _generate_qry_params(self):
        qry_key = self.config['qry'].get('keyword', '')

        # Query Codes
        qry_codes_dir = self.config['io']['dirs']['resc-query-ds-codes']
        qry_codes_fname = collect_fnames_from_folder(qry_codes_dir,
                                                     criteria=[qry_key])
        key = generate_keychain(['qry', 'codes'])
        parameters = {'io.file.' + key: qry_codes_fname[0]}

        # Query Config
        qry_config_dir = self.config['io']['dirs']['resc-query-ds-config']
        qry_config_fname = collect_fnames_from_folder(qry_config_dir,
                                                      criteria=[qry_key])

        key = generate_keychain(['qry', 'config'])
        parameters = {**parameters,
                      'io.file.' + key: qry_config_fname[0]}

        return parameters

    def _generate_mod_params(self):

        mod_config = self.config.get('mod', None)
        child = self.config['child']

        if mod_config is not None:
            mod_type = self.config['mod'].get('type', child)
            mod_key = self.config['mod'].get('keyword', 'default')

            criteria = [mod_type, mod_key]

            # Models (mod)
            mod_dir = self.config['io']['dirs']['resc-models-ds-models']
            mod_criteria = criteria+['pkl']
            mod_fnames = collect_fnames_from_folder(mod_dir,
                                                    criteria=mod_criteria,
                                                    indexed=True)
            key = generate_keychain(['load', 'mod'])
            parameters = {'io.file.' + key: mod_fnames}

            # Model config (mod-config)
            mod_config_dir = self.config['io']['dirs']['resc-models-ds-config']
            mod_config_fname = collect_fnames_from_folder(mod_config_dir,
                                                          criteria=criteria)

            key = generate_keychain(['load', 'mod', 'config'])
            parameters = {**parameters,
                          'io.file.' + key: mod_config_fname[0]}
        else:
            parameters = {}

        return parameters
