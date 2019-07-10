import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from timeit import default_timer

class VersaDummy(object):
    """
    Dummy classifier that can take queries.
    """

    def __init__(self):
        self.dummies = []

        self.s = {'model_data': {'ind_time': -1,
                                 'inf_time': -1}}
        return

    def fit(self, data, **kwargs):

        if isinstance(data, pd.DataFrame):
            data = data.values
        assert isinstance(data ,np.ndarray)

        tick = default_timer()
        nb_rows, nb_atts = data.shape

        self.dummies = [DummyClassifier(**kwargs)
                        for col in range(nb_atts)]

        att_idxs = list(range(nb_atts))
        for targ_idx in att_idxs:
            desc_idx = att_idxs.copy().remove(targ_idx)

            X = data[:, desc_idx]
            Y = data[:, targ_idx]

            self.dummies[targ_idx].fit(X, Y)

        tock = default_timer()
        self.s['model_data']['inf_time'] = tock - tick
        return

    def predict(self, data, targ_idx):
        if isinstance(data, pd.DataFrame):
            data = data.values
        assert isinstance(data ,np.ndarray)
        assert isinstance(targ_idx, list)

        nb_rows, _ = data.shape
        nb_targ = len(targ_idx)

        res = np.zeros((nb_rows, nb_targ))

        tick = default_timer()
        for t_idx, t in enumerate(targ_idx):
            res[:, t_idx] = self.dummies[t].predict(data)

        tock = default_timer()
        self.s['model_data']['inf_time'] = tock - tick

        return res
