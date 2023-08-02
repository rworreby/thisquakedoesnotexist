#!/usr/bin/env python3

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def set_up_folders(run_id, args):
    """
    Output configurations dictionary and create data directory
    """
    output_dir = args.output_dir + "_" + str(run_id)
    model_file = args.model_file

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create diretory for models
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # create directory for figures
    training_dir = os.path.join(output_dir, "training_plots")
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    metrics_dir = os.path.join(output_dir, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    grid_dir = os.path.join(output_dir, "grid_plots")
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)

    fig_dir = os.path.join(output_dir, "figs")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # make a copy of gan.py which contains the model specifications
    model_file_name = model_file.split("/")[-1]
    gan_out = os.path.join(output_dir, model_file_name)
    shutil.copyfile(model_file, gan_out)
    dirs = {
        "output_dir": output_dir,
        "models_dir": models_dir,
        "training_dir": training_dir,
        "fig_dir": fig_dir,
        "metrics_dir": metrics_dir,
        "grid_dir": grid_dir,
    }
    return dirs


# rescale conditional variables
def rescale(v):
    """
    Reascale numpy array to be in the range 0,1
    """
    v_min = v.min()
    v_max = v.max()
    dv = v_max - v_min
    # rescale to [0,1]
    vn = (v - v_min) / dv
    # rescale to [-1,1]
    # vn = 2.0 * vn - 1.0
    return vn


# factory function to rescale variables
def make_maps_scale(v_min, v_max):
    """
    Rescale arrays based on max_v, where min comes from the data
    """

    def to_scale_11(x):
        dv = v_max - v_min
        xn = (x - v_min) / dv
        # rescale to [-1,1]
        xn = 2.0 * xn - 1.0
        return xn

    def to_real(x):
        dv = v_max - v_min
        xr = (x + 1.0) / 2.0
        xr = xr * dv + v_min
        return xr

    # return tuple of functions
    return (to_scale_11, to_real)


class SeisData(object):
    """
    Class to manage seismic data
    """

    # ------- define key class varibles ----------------
    # numpy array (Nobs, 3 , Nt) normalized waveforms
    wfs = None
    # not normalized waveforms (Nobs, 3 , Nt)
    ws = None
    # normalization factors (Nobs, 3 )
    cnorms = None

    # list with pseudo continuous conditional variables scaled to [0,1)
    vc_lst = None
    # dictionary with min value for conditional varible
    vc_max = None
    # dictionary with min value for conditional varible
    vc_min = None

    # list with names for conditional variables, must match the name of the field in the
    # pandas data fram
    v_names = None
    # pandas data frame with raw values for conditional variables
    # has the fields defined in v_names
    df_meta = None
    # pandas dataframe with all conditional variables
    df_meta_all = None

    # -------------------------------------------------

    def __init__(self, data_file, attr_file, batch_size, sample_rate, v_names, isel):
        """
        Loads data and creates the data handler object
        :param data_file: file with waveform data
        :param attr_file: file with labels for waveform
        :param batch_size: batch size to use
        :param sample_rate: sampling rate for waveforms
        :param v_names: List with conditional variable names
        :param isel: set of indices to use
        """
        # store configuration parameters
        self.data_file = data_file
        self.batch_size = batch_size

        if not isinstance(v_names, list):
            assert False, "Please supply names of conditional variables"

        self.v_names = v_names
        # load data
        print("Loading data ...")
        wfs = np.load(data_file)
        print("Loaded samples: ", wfs.shape[0])
        # total number of training samples
        Ntrain = wfs.shape[0]
        self.Ntrain = Ntrain
        # store non normalized waveforms
        self.ws = wfs.copy()
        print("normalizing data ...")
        wfs_norm = np.max(np.abs(wfs), axis=1)  # 2)
        # store normalization facots
        cnorms = wfs_norm.copy()
        self.cnorms = cnorms
        # for braadcasting
        wfs_norm = wfs_norm[:, np.newaxis]
        # apply normalization
        self.wfs = wfs / wfs_norm

        # --- rescale norms -----
        vns_max = np.max(cnorms, axis=0)
        vns_min = np.min(cnorms, axis=0)
        pga_max = vns_max.max()
        pga_min = vns_min.min()
        log_pga_max = np.log10(pga_max)
        log_pga_min = np.log10(pga_min)
        lc_m = np.log10(cnorms)
        fn_to_scale_11, fn_to_real = make_maps_scale(log_pga_min, log_pga_max)
        # create array for rescaled varibles
        ln_cns = np.zeros_like(lc_m)

        ln_cns[:] = fn_to_scale_11(lc_m[:])
        # ln_cns[:,1] = fn_to_scale_11(lc_m[:,1])
        # ln_cns[:,2] = fn_to_scale_11(lc_m[:,2])
        # store variables of interest
        self.fn_to_scale_11 = fn_to_scale_11
        self.fn_to_real = fn_to_real
        self.ln_cns = ln_cns

        # time domain attributes for convinience
        self.sample_rate = sample_rate
        self.dt = 1 / sample_rate

        # load attributes for waveforms
        df = pd.read_csv(attr_file)
        # store All attributes
        self.df_meta_all = df.copy()
        # store pandas dict as attribute
        self.df_meta = df[self.v_names]

        # ----- Configure conditional variables --------------
        # store normalization constant for conditional variables
        self._set_vc_max()
        # set values for conditional variables
        self._init_vcond()

        # partion the dataset
        Nsel = len(isel)

        # ----- sample a fracion of the dataset ------
        self.wfs = self.wfs[isel, :]
        self.ws = self.ws[isel, :]
        self.cnorms = self.cnorms[isel]
        self.ln_cns = self.ln_cns[isel]

        # get a list with the labels
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[isel, :])
        self.vc_lst = vc_b

        # indeces for the waveforms
        self.ix = np.arange(Nsel)
        # numpy array with conditional variables
        # maybe not needed
        self.vc = np.hstack(self.vc_lst)

        self.Ntrain = Nsel
        print("Number selected samples: ", Nsel)

        print("Class init done!")

    def to_real(self, vn, v_name):
        # Rescale variable to its original range
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max - v_min
        # rescale to [v_min, v_max]
        v = vn * dv + v_min
        # rescale to [-1,1]
        # vn = 2.0 * vn - 1.0
        return v

    def to_syn(self, vr, v_name):
        # Rescale variable to the predefined
        # synthetic [0, 1] range
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max - v_min
        # rescale to [0,1]
        vn = (vr - v_min) / dv
        return vn

    def _set_vc_max(self):
        """
        store normalization constant for conditional variables
        """
        self.vc_max = {}
        self.vc_min = {}
        for vname in self.v_names:
            v_max = self.df_meta[vname].max()
            self.vc_max[vname] = v_max
            v_min = self.df_meta[vname].min()
            self.vc_min[vname] = v_min

    def _init_vcond(self):
        """
        Set values for conditional variables using pseudo binning strategy
        each variable is normalized to be in [0,1) and it is assigned the value
        of its bin midpoint
        """

        self.vc_lst = []
        for v_name in self.v_names:
            print("---------", v_name, "-----------")
            print("min " + v_name, self.df_meta[v_name].min())
            print("max " + v_name, self.df_meta[v_name].max())
            # 1. rescale variables to be between 0,1
            v = rescale(self.df_meta[v_name].to_numpy())
            # reshape conditional variables
            vc = np.reshape(v, (v.shape[0], 1))
            print("vc shape", vc.shape)
            # 3. store conditional variable
            self.vc_lst.append(vc)
        ## end method

    def _get_rand_idxs(self):
        """
        Randomly sample data
        :return: sorted random indeces for the batch that is going to be used
        """
        ib = np.random.choice(self.ix, size=self.batch_size, replace=False)
        ib.sort()
        return ib

    def get_rand_batch(self):
        """
        get a random sample from the data with batch_size 3C waveforms
        :return: wfs (numpy.ndarray), cond vars (list)
        """
        ib = self._get_rand_idxs()
        wfs_b = self.wfs[ib, :]
        # get the corresponding normalization constants
        ln_cns_b = self.ln_cns[ib]
        # expand dimension to treat data as a 2-D tensor
        wfs_b = wfs_b[:, :, np.newaxis]
        # get a list with tshe labels
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[ib, :])

        return (wfs_b, ln_cns_b, vc_b)

    def get_rand_cond_v(self):
        """
        Get a random sample of conditional variables
        """
        vc_b = []
        # sample conditional variables at random
        for v in self.vc_lst:
            ii = self._get_rand_idxs()
            vc_b.append(v[ii, :])

        return vc_b

    def get_batch_size(self):
        """
        Get the batch size used
        :return: int = batch size
        """
        return self.batch_size

    def __str__(self):
        return "wfs data shape: " + str(self.wfs.shape)

    def get_Ntrain(self):
        """
        Get total number of training samples in the seismic dataset
        :return: int
        """
        return self.Ntrain

    def get_Nbatches_tot(self):
        """
        get the total number of batches requiered for training: Ntrain/batch_size
        :return: int: number of batches
        """
        Nb_tot = np.floor(self.Ntrain / self.batch_size)
        return int(Nb_tot)
