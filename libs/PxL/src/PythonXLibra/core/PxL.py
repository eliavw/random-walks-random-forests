import json
import os
import numpy as np
import pandas as pd
from timeit import default_timer

from os.path import abspath, dirname

from .run import (generate_monitor, run_process)
from .utils import debug_print

VERBOSITY = 1

LIBRA_INSTALL_FROST = "/home/elia/.opam/system/bin/libra"
LIBRA_INSTALL_JUP = "/cw/dtaijupiter/NoCsBack/dtai/elia/00_Software/.opam/system/bin/libra"
LIBRA_INSTALL_ZISSOU = "/home/zissou/.opam/4.02.3/bin/libra"
PXL_DIR = dirname(dirname(abspath(__file__)))


class PxL(object):
    """
    Python x Libra main class.

    This object acts as some kind of python front-end to the Libra package, and
    tries to follows the scikit-learn interface as much as possible.
    """

    # Constants
    _pxl_dir = PXL_DIR

    # Defaults - Running
    _default_to_fit = 10 * 60 * 60
    _default_to_pred = 2 * 60 * 60
    _default_log_fn = "PxL_log"

    # Defaults - Tmp files
    _default_model_bname = "model"
    _default_model_ext = ".bn"
    _default_evidence_fname = "evidence.csv"
    _default_pred_fn = "pred.csv"

    def __init__(self, cwd=None, log_fname=None, machine=None):

        # Run Params - cwd
        if cwd is not None:
            assert os.path.abspath(cwd)
            self.cwd = cwd
        else:
            self.cwd = os.getcwd()

        self.set_machine(machine=machine)

        # Run Params - log
        self.generate_log_fname(log_fname=log_fname, kind="all")

        # Run Params - Timings
        self.s = {'model_data': {'ind_time': -1.0,
                                 'inf_time': -1.0}}

        self.cfg_fit = self._default_fit_cfg()
        self.cfg_pred = self._default_pred_cfg()
        self.cfg_cvrt = self._default_cvrt_cfg()

        return

    # Logs
    def set_cwd(self, cwd):
        """
        Changing cwd after init requires a change of logs too.

        Parameters
        ----------
        cwd

        Returns
        -------

        """
        self.cwd = cwd
        self.generate_log_fname(kind="all")
        return

    def generate_log_fname(self, log_fname=None, kind="fit"):
        if log_fname is None:
            log_base_name = str(id(self)) + '_' + self._default_log_fn
        else:
            log_base_name = log_fname

        # Ensure cwd if necessary
        if not os.path.isabs(log_base_name):
            log_base_name = os.path.join(self.cwd, log_base_name)

        if kind in {"fit"}:
            self.log_fname_fit = log_base_name + "_fit"
        elif kind in {"predict", "pred"}:
            self.log_fname_predict = log_base_name + "_predict"
        elif kind in {"convert", "cvrt"}:
            self.log_fname_cvrt = log_base_name + "_convert"
        elif kind in {"all"}:
            self.generate_log_fname(log_fname, kind="fit")
            self.generate_log_fname(log_fname, kind="predict")
            self.generate_log_fname(log_fname, kind="cvrt")
        else:
            msg = """
            Did not recognize kind:     {}
            Cannot generate log for this operation.
            """.format(kind)
            raise ValueError(msg)
        return

    # Config defaults
    def _default_fit_cfg(self):

        cfg = {"algo":  "bnlearn",
               'o':      self._default_model_bname + self._default_model_ext}

        cfg['o'] = self._consistent_ext(cfg['o'],
                                        cfg["algo"])

        return cfg

    def _default_pred_cfg(self):

        cfg = {"ev":    self._default_evidence_fname,
               'mo':    self._default_pred_fn}

        return cfg

    def _default_cvrt_cfg(self):

        cfg = {"algo": "dn2mn",
               'o':    self._default_model_bname + self._default_model_ext}

        cfg['o'] = self._consistent_ext(cfg['o'],
                                        cfg["algo"])

        return cfg

    # Core methods
    def fit(self, timeout=None, **kwargs):
        # Set our arguments
        if timeout is None:
            timeout = self._default_to_fit

        # Update config
        self.cfg_fit = {**self.cfg_fit,
                        **kwargs}
        self.cfg_fit = self._ensure_cwd(self.cfg_fit)
        self.cfg_fit['o'] = self._consistent_ext(self.cfg_fit['o'],
                                                 self.cfg_fit["algo"])

        # Run - Prelims
        mon = generate_monitor(self.log_fname_fit, timeout)
        cmd = self._generate_command(kind='fit')

        msg = """
        Generated command: {}
        """.format(cmd)
        debug_print(msg, V=VERBOSITY)

        # Run
        tick = default_timer()
        p = run_process(cmd, monitors=mon)
        tock = default_timer()
        self.s['model_data']['ind_time'] = tock - tick

        try:
            if p != 0: raise ValueError

            self._drop_log(self.log_fname_fit)

            self.cfg_pred['m'] = self.cfg_fit['o']
            self.cfg_cvrt['m'] = self.cfg_fit['o']
            self.cfg_cvrt['i'] = self.cfg_fit['i']
            return p
        except ValueError:
            msg = """
                fit did not go well,
                Returncode:         {}                        
                """.format(p)
            print(msg)
            return p
        except FileNotFoundError as e:
            msg = """
                FileNotFoundError:                                  {}
                Returncode from libra:                              {}                         
                """.format(e.args[-1], p)
            print(msg)
            return p

    def predict(self,
                test_fname,
                miss_idx=None,
                targ_idx=None,
                timeout=None,
                q_idx=None,
                **kwargs):
        """
        Predict the query.

        I.e. MAP-inference.

        Parameters
        ----------
        test_fname
        miss_idx
        targ_idx
        timeout
        q_idx
        kwargs

        Returns
        -------
        """

        # Set our arguments
        if timeout is None:
            timeout = self._default_to_pred

        # Update config
        self.cfg_pred = {**self.cfg_pred,
                         **kwargs}
        self.cfg_pred = self._ensure_cwd(self.cfg_pred)

        # Run - Prelims
        if q_idx is not None:
            # Alterations to file in case of a query index.
            self.log_fname_predict = self._alter_fname_for_q_idx(self.log_fname_predict, q_idx)
            self.cfg_pred["ev"] = self._alter_fname_for_q_idx(self.cfg_pred["ev"], q_idx)
            self.cfg_pred["mo"] = self._alter_fname_for_q_idx(self.cfg_pred["mo"], q_idx)

        self._generate_evidence(test_fname, miss_idx, targ_idx) # Create evidence

        msg = """
        pred log at:        {}
        """.format(self.log_fname_predict)
        debug_print(msg, V=VERBOSITY)
        mon = generate_monitor(self.log_fname_predict, timeout)
        cmd = self._generate_command(kind="predict")

        msg = """
        Generated command: {}
        """.format(cmd)
        debug_print(msg, V=VERBOSITY)

        # Run
        tick = default_timer()
        p = run_process(cmd, monitors=mon)
        tock = default_timer()
        self.s['model_data']['inf_time'] = tock - tick

        try:
            if p != 0: raise ValueError
            self._drop_log(self.log_fname_predict)

            result = self._read_and_filter_output(self.cfg_pred["mo"], targ_idx)
            os.remove(self.cfg_pred["mo"])
            os.remove(self.cfg_pred["ev"])

            return result.values
        except ValueError:
            msg = """
            prediction did not go well,
            Returncode:             {}                        
            """.format(p)
            print(msg)
            return p
        except FileNotFoundError as e:
            msg = """
            FileNotFoundError:                                  {}
            Returncode from libra:                              {}                         
            """.format(e.args[-1], p)
            print(msg)
            return p

    def convert(self, timeout=None, **kwargs):
        # Set our arguments
        if timeout is None:
            timeout = self._default_to_fit

        # Update config
        self.cfg_cvrt = {**self.cfg_cvrt,
                         **kwargs}
        self.cfg_cvrt = self._ensure_cwd(self.cfg_cvrt)

        # Run - Prelims
        mon = generate_monitor(self.log_fname_cvrt, timeout)
        cmd = self._generate_command(kind="convert")

        msg = """
        Generated command: {}
        """.format(cmd)
        debug_print(msg, V=VERBOSITY)

        # Run
        tick = default_timer()
        p = run_process(cmd, monitors=mon)
        tock = default_timer()
        self.s['model_data']['ind_time'] = tock - tick

        try:
            if p != 0: raise ValueError

            self._drop_log(self.log_fname_cvrt)

            self.cfg_pred['m'] = self.cfg_cvrt['o']
            return p
        except ValueError:
            msg = """
                fit did not go well,
                Returncode:         {}                        
                """.format(p)
            print(msg)
            return p
        except FileNotFoundError as e:
            msg = """
                FileNotFoundError:                                  {}
                Returncode from libra:                              {}                         
                """.format(e.args[-1], p)
            print(msg)
            return p

    # Helpers
    def set_machine(self, machine=None):
        """
        Setting the machine parameter also can happen outside init.

        Parameters
        ----------
        machine

        Returns
        -------

        """

        if machine in {"jup", "jupiter"}:
            self.libra_install = LIBRA_INSTALL_JUP
        elif machine in {"belafonte", "laptop", "zissou"}:
            self.libra_install = LIBRA_INSTALL_ZISSOU
        else:
            self.libra_install = LIBRA_INSTALL_FROST
        return

    def _ensure_cwd(self, cfg, keys=None):
        """
        Alter cfg.

        Note, since cfg is a dict, and python, this ALTERS the dict.

        Parameters
        ----------
        cfg
        keys

        Returns
        -------

        """

        if keys is None:
            keys = ['i', 'm', 'o', "ev", 'q', "mo"]

        for k in keys:
            if k in cfg.keys():
                fname = cfg[k]
                if not os.path.isabs(fname):
                    cfg[k] = os.path.join(self.cwd, fname)

        return cfg

    def _generate_command(self, kind="fit"):

        if kind in {"fit"}:
            cfg = self.cfg_fit
        elif kind in {"predict"}:
            cfg = self.cfg_pred
        elif kind in {"convert"}:
            cfg = self.cfg_cvrt
        else:
            msg = """
            Did not recognize kind:     {}
            Accepted kinds are [fit", "predict"]
            """.format(kind)
            raise ValueError(msg)

        cmd = [self.libra_install, cfg.get("algo")]

        for k, v in cfg.items():

            # Some preprocessing
            flag = '-' + k

            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # Numbers have to be stringed.
                v = str(v)

            if k in {"algo"}:
                pass
            elif isinstance(v, bool):
                # Booleans are simply flags without value.
                if v:
                    cmd.append(flag)
            else:
                cmd.append(flag)
                cmd.append(v)

        return cmd

    def _generate_evidence(self,
                           test_fname,
                           miss_idx,
                           targ_idx,
                           missing_symbol='*'):

        # Adapt relative paths to cwd
        cfg = self._ensure_cwd(self.cfg_pred)
        evidence_fname = cfg["ev"]

        if not os.path.isabs(test_fname):
            test_fname = os.path.join(self.cwd, test_fname)

        unknown_idx = miss_idx + targ_idx

        df = pd.read_csv(test_fname, header=None)
        df.iloc[:, unknown_idx] = missing_symbol
        df.to_csv(evidence_fname, header=None, index=None)
        return

    @staticmethod
    def _alter_fname_for_q_idx(fname, q_idx):
        assert isinstance(q_idx, int)
        q_idx_appendix = "_Q" + str(q_idx).zfill(4)
        base_fname, ext = os.path.splitext(fname)

        if "_Q" in fname:
            base, old_appendix = base_fname.split("_Q")
            base_fname = base + q_idx_appendix
        else:
            base_fname += q_idx_appendix

        return base_fname + ext

    @staticmethod
    def _consistent_ext(fname, algo):
        algo_ext_map = {"cl":        ".bn",
                        "bnlearn":   ".bn",
                        "dnlearn":   ".dn",
                        "dnboost":   ".dn",
                        "dn2mn":     ".mn"}

        base, _ = os.path.splitext(fname)
        fname = base + algo_ext_map.get(algo, None)

        return fname

    @staticmethod
    def _read_and_filter_output(out_fname, targ_idx):
        assert isinstance(targ_idx, list)
        df = pd.read_csv(out_fname, header=None)
        df = df.iloc[:, targ_idx]
        return df

    @staticmethod
    def _drop_log(log_fname):

        log_dir = os.path.dirname(log_fname)
        log_bfn = os.path.basename(log_fname)

        success_log_fnames = [f for f in os.listdir(log_dir)
                              if log_bfn in f
                              if "success" in f]

        msg = """
        log_fname:          {}
        log_dir:            {}
        log_bfn:            {}
        success_log_fnames: {}
        """.format(log_fname, log_dir, log_bfn, success_log_fnames)
        debug_print(msg, V=VERBOSITY)

        for f in success_log_fnames:
            os.remove(os.path.join(log_dir, f))

        return


