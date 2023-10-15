# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_waves_1C(ws, dt, t_max=50.0, ylim=None, fig_file=None, Np=12, fig_size = (6,9), stitle=None, show_fig=False):
    """
        Make plot of waves
        shape of the data has the form -> (Nobs, Nt)
        :param ws_p: numpy.ndarray of shape (Nobs, Nt)
        :param Np: number of 3C waves to show
        :param color: Color of the plot
        :param t_max: max time for the plot in sec
        :return: show a panel with the waves

        """
    Nt = int(t_max / dt)
    if ws.shape[0] < Np:
        # in case there are less observations than points to plot
        Np = ws.shape[0]

    ws_p = ws[:Np, :]
    
    # select waves to plot
    tt = dt * np.arange(Nt)
    plt.figure()
    fig, axs = plt.subplots(Np, 1, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.3, 'wspace': 0},
                            figsize=fig_size)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    for ik in range(Np):
        wf = ws_p[ik,:]
        axs[ik].plot(tt, wf, lw=0.5)
        if ylim is not None:
            axs[ik].set_ylim(ylim)

    for ax in axs.flat:
        ax.label_outer()

    if stitle is not None:
        fig.suptitle(stitle, fontsize=12)

    if show_fig:
        fig.show()

    if fig_file is not None:
        fformat = fig_file.split('.')[-1]
        print('saving:', fformat)
        fig.savefig(fig_file, format=fformat)
        if not show_fig:
            plt.close()
    else:
        fig.show()
    plt.close()


# ---- binning functions ------

def make_bins(v, bins):
    if isinstance(bins, int):
        b = np.linspace(np.min(v), np.max(v), bins + 1)
        b[-1] = np.inf
        return b
    elif isinstance(bins, list):
        b = np.array([np.inf if b_ == 'inf' else b_ for b_ in bins])
        return b
    else:
        assert False, "Bins must either be specified as a list or an interger."


def make_ibins(v, nbins):
    """
    Create array with bin edges
    """
    # make sure nbins is interger
    assert isinstance(nbins, int), "nbins must be an interger."

    b_max = np.max(v)
    b_min = np.min(v)
    dbin = (b_max - b_min) / nbins
    # slightly increase b_max so that largest value in array gets assigned
    # to nbins -1 when using np.digitize
    b_max = b_max + 0.01 * dbin
    b = np.linspace(b_min, b_max, nbins + 1)
    return b


# get mid point for bins
def get_bins_midpoint(bins):
    Nb = len(bins) - 1
    cb = np.zeros((Nb), dtype=np.float32)
    for i in range(Nb):
        cb[i] = (bins[i] + bins[i + 1]) / 2.0
    return cb


# rescale conditional variables
def rescale(v):
    """
    Reascale numpy array to be in the range 0,1
    """
    v_min = v.min()
    v_max = v.max()
    dv = v_max-v_min
    # rescale to [0,1]
    vn = (v-v_min) / dv
    # rescale to [-1,1]
    #vn = 2.0 * vn - 1.0
    return vn



class SeisData(object):
    """
    Class to manage seismic data
    """
    # ------- define key class varibles ----------------
    # numpy array (Nobs, 3 , Nt)
    wfs = None
    vc_lst = None
    iv_lst = None
    vc_max = None
    vc_min = None

    # list with names for conditional variables, must match the name of the field in the
    # pandas data fram
    v_names = None
    # pandas data frame with raw values for conditional variables
    # has the fields defined in v_names
    df_meta = None
    # dict with bin edges for each conditional variable
    ebins = None
    # dict with bin mid points for each conditional variable
    bin_midpts = None

    # -------------------------------------------------

    def __init__(self, data_file, attr_file, batch_size, sample_rate, v_names, nbins_d, isel):
        """
        Loads data and creates the data handler object
        :param data_file: file with waveform data
        :param attr_file: file with labels for waveform
        :param batch_size: batch size to use
        :param sample_rate: sampling rate for waveforms
        :param v_names: List with conditional variable names
        :param nbins_d: Dictionary with definition of bins for conditional variabels
        :param isel: set of indices to use
        """
        # store configuration parameters
        self.data_file = data_file
        self.batch_size = batch_size

        if not isinstance(v_names, list):
            assert False, "Please supply names of conditional variables"

        self.v_names = v_names
        self.nbins_d = nbins_d
        # load data
        print('Loading data ...')
        wfs = np.load(data_file)
        print('Loaded samples: ', wfs.shape[0])
        # total number of training samples
        Ntrain = wfs.shape[0]
        self.Ntrain = Ntrain
        print('normalizing data ...')

        # Robin:
        # Changed axis from 2 to 1
        wfs_norm = np.max(np.abs(wfs), axis=1)

        # for braadcasting
        wfs_norm = wfs_norm[:, np.newaxis]
        # apply normalization
        self.wfs = wfs / wfs_norm

        # time domain attributes for convinience
        self.sample_rate = sample_rate
        self.dt = 1 / sample_rate

        # load attributes for waveforms
        df = pd.read_csv(attr_file)
        # store pandas dict as attribute
        self.df_meta = df[self.v_names]

        # ----- Configure conditional variables --------------
        # store normalization constant for conditional variables
        self._set_vc_max()
        # initialize binning configuration
        self._init_bins()
        # set values for conditional variables
        self._init_vcond()


        # partion the dataset
        Nsel = len(isel)

        # sample a fracion of the dataset
        # Robin:
        # Changed dim down from 3 to 1
        self.wfs = self.wfs[isel]
        # self.wfs = self.wfs[isel, :, :]

        # get a list with the labels
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[isel, :])
        self.vc_lst = vc_b
        # get conditional indexes
        iv_b = []
        for iv in self.iv_lst:
            iv_b.append(iv[isel, :])
        self.iv_lst = iv_b

        # indeces for the waveforms
        self.ix = np.arange(Nsel)
        # numpy array with conditional variable
        # may be a bug here
        #self.vc = np.hstack(self.iv_lst)
        self.vc = np.hstack(self.vc_lst)

        # numpy array with conditional indexes
        self.iv_s = np.hstack(self.iv_lst)
        self.Ntrain = Nsel
        print('Number selected samples: ', Nsel)

        print('Class init done!')

    def to_real(self, vn, v_name):
        # Rescale variable to its original range
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max - v_min
        # rescale to [v_min, v_max]
        v = vn*dv + v_min
        # rescale to [-1,1]
        #vn = 2.0 * vn - 1.0
        return v

    def to_syn(self, vr, v_name):
        # Rescale variable to the predefined
        # synthetic [0, 1] range
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max-v_min
        # rescale to [0,1]
        vn = (vr-v_min) / dv
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

    def _init_bins(self):
        """
        init dictionaries with bin edges and bin midpoints
        """
        self.ebins = {}
        self.bin_midpts = {}
        for v_name in self.v_names:
            # 1. rescale
            v = rescale(self.df_meta[v_name].to_numpy())
            # 2. define bin edges
            self.ebins[v_name] = make_ibins(v, self.nbins_d[v_name])
            # 3. define bin mid points
            self.bin_midpts[v_name] = get_bins_midpoint(self.ebins[v_name])
        ## end method

    def _init_vcond(self):
        """
        Set values for conditional variables using pseudo binning strategy
        each variable is normalized to be in [0,1) and it is assigned the value
        of its bin midpoint
        """
        # make sure configuration variables are initialized
        assert isinstance(self.ebins, dict), "Bins must be initialized"
        assert isinstance(self.bin_midpts, dict), "Bins must be initialized"


        self.iv_lst = []
        self.vc_lst = []
        for v_name in self.v_names:
            # 1. rescale variables to be between 0,1
            v = rescale(self.df_meta[v_name].to_numpy())
            # 2. get bin edges
            ebins = self.ebins[v_name]
            print('---------', v_name, '-----------')
            #print('bins ' + v_name + ': ', ebins)
            print('min ' + v_name, self.df_meta[v_name].min())
            print('max ' + v_name, self.df_meta[v_name].max())

            # 3 .find the correct bin index for each value in the array
            iv = np.digitize(v, ebins) - 1
            # reshape into vertical vector
            iv = np.reshape(iv, (iv.shape[0], 1))
            # store the bin index
            self.iv_lst.append(iv)
            # 4. get bin midpoints
            mid_bins = self.bin_midpts[v_name]
            # 5.
            # extract the midpoint using the bin index and the mid_bins array
            # the function fn is applied to each element in iv array
            fn = lambda ix: mid_bins[ix]
            vc = np.apply_along_axis(fn, 0, iv)
            print('vc shape', vc.shape)
            # store conditional variable
            self.vc_lst.append(vc)
        ## end method

    def _get_condv(self):
        """
        return numpy array with conditional variables
        """
        return self.iv_s

    def get_batch_vcond(self, iv_lst, Np=10):
        """
        get a batch random batch of data data satisfies the conditional variables
        :return: waveforms
        """
        i_sel = self.iv_s[:, 0] == iv_lst[0]
        Nv = len(iv_lst)
        for ii in range(1, Nv):
            i_sel = i_sel & (self.iv_s[:, ii] == iv_lst[ii])

        assert i_sel.sum() > 0, "No Elements meet the conditions"
        
        print('batch_vcond shape', ws.shape)
        if ws.shape[0] > Np:
            # randomly select a batch of waves
            i_out = np.random.choice(Np, size=(Np,), replace=False)
            i_out.sort()
            return ws[i_out, :, :]
        else:
            return ws

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
        # expand dimension to treat data as a 2-D tensor
        # Robin:
        # Removed one axis -> 3C to 1C
        wfs_b = wfs_b[:, np.newaxis]
        # get a list with the labels
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[ib, :])

        # breakpoint()

        return (wfs_b, vc_b)

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
        return 'wfs data shape: ' + str(self.wfs.shape)

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

    def plot_waves_rand_3C(self, t_max=20.0, Np=10, color='C0', ):
        """
        Make plot of waves
        shape of the data has the form -> (Nobs, 3, Nt)
        :param t_max: max time for the plot in sec
        :param Np: number of 3C waves to show
        :param color: Color of the plot
        :return: show a panel with the waves
        """
        dt = self.dt
        # get a batch of waves at random
        (ws, _) = self.get_rand_batch()
        plot_waves_3C(ws, dt, t_max, Np=Np, color=color)

# %%
# # ---- Testing ----
# data_file = '/Users/manuel/seisML/raw_data/mini_downsamp_5x_all.npy'
# attr_file = '/Users/manuel/seisML/raw_data/mini_wforms_table.csv'
# Nbatch = 32
# bins_dict = {
#     'dist': 4,
#     'mag': [3.0, 4.0, 5.0, 6.0, 'inf'],
#
# }
# v_names = ['dist', 'mag']
# s_dat = SeisData(data_file, attr_file=attr_file, batch_size=Nbatch, sample_rate=20.0, v_names=v_names, bins_d=bins_dict)
#
# print(s_dat._get_rand_idxs())
#
# # get a random sample
# (wfs, i_vc) = s_dat.get_rand_batch()
# print('shape:', wfs.shape)
# s_dat.plot_waves_3C(Np=4, color='C1')
