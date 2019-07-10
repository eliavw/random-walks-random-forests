import ctypes as ctypes
import os
import numpy as np
from timeit import default_timer

from numpy.ctypeslib import ndpointer


try:
    from os.path import dirname
    # Custom imports
    this_dir = dirname(__file__)
    lib_PxS = ctypes.cdll.LoadLibrary(os.path.join(this_dir, 'libPxS.so'))
except:
    msg="""
    Failed to load .so C-library to interface with BayesFusion.
    """
    print(msg)


class PxS(object):
    """
    SMILE x Python


    Scikit-Learn style Wrapper for BayesFusion SMILE engine.
    """

    # region Bindings with C-wrapper
    # Existential functions
    _init_dataset = lib_PxS.init_dataset
    _init_dataset.restype = ctypes.c_void_p

    _init_model = lib_PxS.init_model
    _init_model.restype = ctypes.c_void_p

    _init_validator = lib_PxS.init_validator
    _init_validator.argtypes = [ctypes.c_void_p,
                                ctypes.c_void_p,
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ctypes.c_size_t,
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ctypes.c_size_t]
    _init_validator.restype = ctypes.c_void_p

    _drop_thing = lib_PxS.drop_thing
    _drop_thing.argtypes = [ctypes.c_void_p]

    _copy_dataset = lib_PxS.copy_dataset
    _copy_dataset.argtypes = [ctypes.c_void_p]
    _copy_dataset.restype = ctypes.c_void_p

    # Dataset-related
    _read_file = lib_PxS.read_file
    _read_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    _write_file = lib_PxS.write_file
    _write_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    _nb_rows = lib_PxS.nb_rows
    _nb_rows.argtypes = [ctypes.c_void_p]
    _nb_rows.restype = ctypes.c_int

    _nb_atts = lib_PxS.nb_atts
    _nb_atts.argtypes = [ctypes.c_void_p]
    _nb_atts.restype = ctypes.c_int

    _has_missing_data = lib_PxS.has_missing_data
    _has_missing_data.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _has_missing_data.restype = ctypes.c_bool

    # Model-related
    _inspect_model = lib_PxS.inspect_model
    _inspect_model.argtypes = [ctypes.c_void_p]

    # Validator-related
    _is_class_node = lib_PxS.is_class_node
    _is_class_node.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _is_class_node.restype = ctypes.c_bool

    _get_validator_ds = lib_PxS.get_validator_ds
    _get_validator_ds.argtypes = [ctypes.c_void_p]
    _get_validator_ds.restype = ctypes.c_void_p

    _get_validator_ds_result = lib_PxS.get_validator_ds_result
    _get_validator_ds_result.argtypes = [ctypes.c_void_p]
    _get_validator_ds_result.restype = ctypes.c_void_p

    # Core functionalities
    _fit = lib_PxS.fit
    _fit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _fit.restype = ctypes.c_int

    _predict = lib_PxS.predict
    _predict.restype = None
    _predict.argtypes = [ctypes.c_void_p,
                         ctypes.c_int,
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         ctypes.c_size_t]
    # endregion

    def __init__(self):
        self.train = ctypes.c_void_p(self._init_dataset())
        self.test = ctypes.c_void_p(self._init_dataset())
        self.val_ds = ctypes.c_void_p(self._init_dataset())
        self.model = ctypes.c_void_p(self._init_model())
        self.s = {}
        self.s['model_data'] = {'ind_time': -1.0,
                                'inf_time': -1.0}
        self.is_del = False

        # All this is connected to the query!
        self.validator = None
        return

    def __del__(self):
        # Explicitly delete things
        if not self.is_del:
            msg = """
            Deleting members now.
            """
            print(msg)
            self._drop_thing(self.train)
            self._drop_thing(self.test)
            self._drop_thing(self.val_ds)
            self._drop_thing(self.model)

            if self.validator is not None:
                self._drop_thing(self.validator)
            self.is_del = True
        else:
            msg = """
            Double delete attempt
            """
            raise ValueError(msg)

        return

    # Core
    def fit(self, train_fname, **kwargs):
        assert isinstance(self.model, ctypes.c_void_p)

        self.read_data(train_fname, kind='train')
        tick = default_timer()
        res = self._fit(self.train, self.model);  # Modifies the model
        tock = default_timer()

        self.s['model_data']['ind_time'] = tock-tick

        self.inspect_model()

        return res

    def predict(self, test_fname, miss_idx=None, targ_idx=None, **kwargs):
        # Preliminaries
        assert isinstance(miss_idx, np.ndarray)
        assert isinstance(targ_idx, np.ndarray)
        assert len(targ_idx) > 0

        # Actions
        self.read_data(test_fname, kind='test')
        tick = default_timer()
        self.make_validator(miss_idx, targ_idx)

        # Outputs
        nb_rows_val = self.nb_rows(kind='val')
        targ_idx_val = self.targ_idx()

        msg = """
        Targets of the validator: {}
        """.format(targ_idx_val)

        if len(targ_idx_val) != len(targ_idx):
            msg = """
            Somehow the query did not get properly communicated to the validator!
            """
            raise BaseException(msg)

        pred_all = np.zeros((nb_rows_val, len(targ_idx_val)), dtype=np.int32)

        # This better come from the validation set.
        for count, att_idx in enumerate(targ_idx_val):
            print("Predicting attribute: {}".format(att_idx))
            pred = self.predict_single_att(att_idx)
            pred_all[:, count] = pred

        tock = default_timer()
        self.s['model_data']['inf_time'] = tock - tick

        return pred_all

    # Introspection
    def inspect_model(self):
        self._inspect_model(self.model)
        return

    def nb_atts(self, kind='train'):
        """
        Get number of attributes of a dataset.
        """

        ds = {'train':  self.train,
              'test':   self.test,
              'val':    self.val_ds}

        assert isinstance(ds[kind], ctypes.c_void_p)
        res = self._nb_atts(ds[kind])

        return res

    def nb_rows(self, kind='train'):
        """
        Get number of rows of a dataset.
        """

        ds = {'train':  self.train,
              'test':   self.test,
              'val':    self.val_ds}

        assert isinstance(ds[kind], ctypes.c_void_p)
        res = self._nb_rows(ds[kind])

        return res

    def shape(self, kind='train'):
        return self.nb_atts(kind=kind), self.nb_rows(kind=kind)

    def has_missing_data(self, att_idx, kind='train'):

        ds = {'train':  self.train,
              'test':   self.test,
              'val':    self.val_ds}

        assert isinstance(ds[kind], ctypes.c_void_p)

        return self._has_missing_data(ds[kind], att_idx)

    def is_class_node(self, att_idx):
        """
        Check if an attribute is a target attribute.

        This means checking whether or not the validator regards this attribute
        as being a `classnode` or not.

        Parameters
        ----------
        att_idx

        Returns
        -------

        """

        assert isinstance(self.validator, ctypes.c_void_p)

        return self._is_class_node(self.validator, att_idx)

    def targ_idx(self):
        """
        Extract the target attributes.

        Since the target attributes are related to the query, and the query is
        handled by the validator (and its corresponding dataset), this
        information needs to be extracted from the validator.

        Returns
        -------

        """
        nb_atts = self.nb_atts(kind='test')
        targ_idx = [idx for idx in range(nb_atts)
                    if self.is_class_node(idx)]
        return targ_idx

    def miss_idx(self, ds='val'):
        """
        Extract the missing attributes.

        Typically, this is query related,a nd thus needs to be extracted from
        the validator dataset.

        Parameters
        ----------
        ds

        Returns
        -------

        """
        nb_atts = self.nb_atts(kind='test')

        miss_idx = [idx for idx in range(nb_atts)
                    if self.has_missing_data(idx, kind=ds)]
        return miss_idx

    # IO
    def read_data(self, fname, kind='train'):

        # Preliminaries
        ds = {'train':  self.train,
              'test':   self.test}
        assert isinstance(ds[kind], ctypes.c_void_p)
        assert isinstance(fname, str)
        fname = ctypes.c_char_p(fname.encode('utf-8'))

        # Actions
        self._read_file(ds[kind], fname)  # Does not return anything, but changes underlying ds object.

        return

    def write_data(self, ds, fname='out.csv'):
        assert isinstance(ds, ctypes.c_void_p)
        assert isinstance(fname, str)
        fname = ctypes.c_char_p(fname.encode('utf-8'))

        self._write_file(ds, fname)
        return

    # Helpers
    def make_validator(self, miss_idx, targ_idx):
        # Preliminaries
        assert isinstance(self.test, ctypes.c_void_p)
        assert isinstance(self.model, ctypes.c_void_p)
        assert self.nb_rows(kind='test') > 0 and self.nb_atts(kind='test') > 0
        assert isinstance(miss_idx, np.ndarray)
        assert isinstance(targ_idx, np.ndarray)

        self._drop_thing(self.val_ds)
        if self.validator is not None:
            self._drop_thing(self.validator)

        # Actions
        miss_idx = miss_idx.astype(np.int32)
        targ_idx = targ_idx.astype(np.int32)
        miss_size = miss_idx.size
        targ_size = targ_idx.size

        self.val_ds = ctypes.c_void_p(self._copy_dataset(self.test))
        val = self._init_validator(self.val_ds,
                                   self.model,
                                   miss_idx,
                                   miss_size,
                                   targ_idx,
                                   targ_size)

        self.validator = ctypes.c_void_p(val)
        return

    def get_validator_result_ds(self):
        return ctypes.c_void_p(self._get_validator_ds_result(self.validator))

    def predict_single_att(self, att_idx):

        nb_rows = self.nb_rows(kind='test')
        assert nb_rows > 0
        pred = np.zeros((1, nb_rows), dtype=np.int32)
        pred_size = pred.size

        self._predict(self.validator, att_idx, pred, pred_size)

        return pred
