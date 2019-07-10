import json

from inspect import signature

from ..utils.filesystem import make_dname
from ..utils.extra import (debug_print,
                           generate_keychain)
from ..utils.config import (add_hierarchical_kwargs_to_dict,
                            filter_kwargs_prefix)
from ..utils.run import run_script

VERBOSITY = 0


class Runner(object):
    """
    A runner object runs a script.

    A runner object can:
        1. Generate a config file for the script
        2. Run the script
        3. Generate a command that runs the script

    Why number 3? Well, for parallel execution this can be handy. Instead of immediately executing
    its script, it just can tell you exactly what it would do, if it were to be executed right away.
    """

    def __init__(self, script=None, config=None, log=None, **kwargs):

        self.config = config
        self.log = log
        self.name = self.__class__.__name__
        self.prods = {}
        self.script = script
        return

    # Config
    def make_config(self, **kwargs):
        self.config = {}

        # Create a default configuration dictionary
        self.default_config(**kwargs)

        # Update the configuration dictionary based on kwargs
        self.update_config(**kwargs)
        return

    def default_config(self, **kwargs):
        pass

    def update_config(self,
                      kind=None,
                      hierarchical=False,
                      overwrite=True,
                      **kwargs):
        """
        Update config based on passed kwargs.

        Two main modes:
            1. hierarchical
                Kwargs are converted into a multi-level dictionary
            2. flat
                Kwargs are passed unaltered

        Parameters
        ----------
        kind: str
            Kind of config passed
        hierarchical: bool
            Flag whether or not we are in hierarchical mode
        overwrite: bool
            Flag whether settings can be overwritten
        kwargs: dict
            A dict with the parameters that need to be included in the settings.

        Returns
        -------

        """

        # Extract
        if kind is None:
            config = self.config
        else:
            kwargs = filter_kwargs_prefix(kind, **kwargs)
            config = self.config.get(kind, {})

        # Hierarchical and Overwrite
        if hierarchical:
            config = add_hierarchical_kwargs_to_dict(config,
                                                     **kwargs)
        else:
            if overwrite:
                config = kwargs
            else:
                config = {**config,
                          **kwargs}

        # Insert
        if kind is None:
            self.config = config
        else:
            self.config[kind] = config

        return

    def save_config(self):
        """
        Save the configuration file.

        The location where it needs to be saved is indicated in the config
        file itself.

        Returns
        -------

        """

        # Inputs
        config = self.config
        config_fname = self.config['io']['file']['config']

        # Actions
        with open(config_fname, 'w') as f:
            json.dump(config, f, indent=4)

        return config_fname

    # Run
    def run(self, **kwargs):

        head_tuple, tuple_list = self.generate_commands(**kwargs)
        assert isinstance(tuple_list, list)

        for param_tuple in tuple_list:
            param_dict = {k: v for k,v in zip(head_tuple, param_tuple)}

            # Use introspection for robust binding
            sig = signature(run_script)
            ba = sig.bind(**param_dict)

            run_script(*ba.args, **ba.kwargs)

        return

    def generate_commands(self, **kwargs):
        return None, None

    # Retrieval
    def get_fname(self, name):
        return self.config['io']['file'][name]

    # Default configs
    def default_io_config_dirs(self, root_dir=None, dataset=None, idx=None):

        d = self._default_io_config_dirs_static(root_dir=root_dir)

        if dataset is not None:
            d = self._default_io_config_dirs_dataset(d, dataset)
            d = self._default_qry_config(d)
            d = self._default_mod_config(d)

        if idx is not None:
            d = self._default_io_config_dirs_idx(d, idx)

        msg = """
        Keys generated so far in io_config_dirs: {}
        """.format(d.keys())
        debug_print(msg, V=VERBOSITY)

        return d

    # Config helpers
    @staticmethod
    def _default_io_config_dirs_static(root_dir=None):
        """
        Immutable part of the filesystem dict.

        Parameters
        ----------
        root_dir: str
            Full path to the root directory of the experiment.

        Returns
        -------

        """

        d = {}

        # Level 0
        for k in {'root'}:
            d[k] = make_dname() if root_dir is None else root_dir

        # Level 1
        for k in {'cmd', 'prod', 'resc'}:
            d[k] = make_dname(name=k, parent_dir=d['root'])

        # Level 2
        for k in {'data', 'models', 'query'}:
            key = generate_keychain(['resc', k])
            d[key] = make_dname(name=k, parent_dir=d['resc'])

        # N.B.: Overwrite resc-data to its subfolder tidy
        for k in {'tidy'}:
            parent_key = generate_keychain(['resc', 'data'])
            d[parent_key] = make_dname(name=k, parent_dir=d[parent_key])

        return d

    @staticmethod
    def _default_io_config_dirs_dataset(d, dataset):
        """
        Add dataset-specific part to filesystem dict.

        Concretely, this subdivision happens only in the resources (resc)
        directory.


        Parameters
        ----------
        d: dict
            Existing filesystem dictionary
        dataset: str
            Name of the dataset for which we generate the folders.

        Returns
        -------

        """

        # In resc.
        for k in {'data', 'models', 'query'}:
            parent_key = generate_keychain(['resc', k])
            key = generate_keychain([parent_key, 'ds'])
            d[key] = make_dname(name=dataset, parent_dir=d[parent_key])

        return d

    @staticmethod
    def _default_qry_config(d):
        """
        Add default query dirs to the dict.

        Parameters
        ----------
        d: dict
            Existing filesystem dictionary

        Returns
        -------

        """
        for k in {'config', 'codes'}:
            parent_key= generate_keychain(['resc', 'query', 'ds'])
            key = generate_keychain([parent_key, k])
            d[key] = make_dname(name=k, parent_dir=d[parent_key])

        return d

    @staticmethod
    def _default_mod_config(d):
        """
        Add default query dirs to the dict.

        Parameters
        ----------
        d: dict
            Existing filesystem dictionary

        Returns
        -------

        """
        for k in {'config', 'models'}:
            parent_key= generate_keychain(['resc', 'models', 'ds'])
            key = generate_keychain([parent_key, k])
            d[key] = make_dname(name=k, parent_dir=d[parent_key])

        return d

    def _default_io_config_dirs_idx(self, d, idx=None):
        """
        Adapt the default filesystem dict for this specific run.

        Parameters
        ----------
        d
        idx

        Returns
        -------

        """

        # N.B.: Overwrite prod to the subfolder corresponding to the appropriate idx
        for k in {self.name}:
            parent_key = 'prod'
            d[parent_key] = make_dname(name=k, parent_dir=d[parent_key], id=idx)

        # For each product-category, create the appropriate subfolder
        for k in self.prods:
            key = generate_keychain(['prod', k])
            d[key] = make_dname(name=k, parent_dir=d['prod'])

        return d
