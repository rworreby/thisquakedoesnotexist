import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def make_ibins(v, nbins):
    """
    :param v: data
    :param nbins: int, number of bins
    :return: array bins edges
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


def make_vc_bins(v, bins, n_decimals=2):
    """

    :param v: numpy array with data
    :param bins: array with bin edges
    :param n_decimals: decimals
    :return: vc, m_bins, cnt_bins
    """
    # 1. find the correct bin index for each value in the array
    iv = np.digitize(v, bins) - 1
    # get average values and counts
    Nb= len(bins)-1
    m_bins = np.zeros((Nb),dtype=np.float)
    cnt_bins = np.zeros((Nb),dtype=np.float)

    for ib in range(Nb):
        ix = iv==ib
        m_bins[ib] = np.mean(v[ix])
        cnt_bins[ib] = ix.sum()

    print("BINS CNT", cnt_bins)
    # round number
    m_bins = np.around(m_bins, decimals=n_decimals)
    print("BINS AVE:", m_bins)
    print("Total elems: ", np.sum(cnt_bins))

    fn = lambda ix: m_bins[ix]
    vc = np.apply_along_axis(fn, 0, iv)
    print('vc shape', vc.shape)
    return (vc, m_bins, cnt_bins)


def make_rescale(v):
    """
    Reascale numpy array to be in the range 0,1
    """
    v_min = v.min()
    v_max = v.max()
    def rescale(x):
        dv = v_max-v_min
        # rescale to [0,1]
        xn = (x-v_min) / dv
        return xn
    # rescale to [-1,1]
    #vn = 2.0 * vn - 1.0
    return rescale


class WaveDatasetDist(Dataset):
    def __init__(self, data_file, attr_file, ndist_bins, dt=0.04):
        # and labels
        x_data = np.load(data_file)
        df = pd.read_csv(attr_file)
        print('Shape attribute file: ', df.shape)

        print('normalizing data ...')
        wfs_norm = np.max(np.abs(x_data), axis=2)
        # for braadcasting
        wfs_norm = wfs_norm[:, :, np.newaxis]
        # apply normalization
        x_data = x_data / wfs_norm
        # keep only second horizontal componenet
        x_data = x_data[:,1,:]
        print("x_data shape: ", x_data.shape)
        print("x_data min: ", x_data.min())
        print("x_data max: ", x_data.max())

        # -------------- set input variables ----------------
        self.df = df
        # create scaling functions
        dist = np.array(df['dist']).reshape(-1,1)
        #mag = np.array(df['mag']).reshape(-1,1)
        # sampling rate
        self.dt = dt

        # get functions
        self.fn_dist_scale = make_rescale(dist)
        #self.fn_mag_scale = make_rescale(mag)

        # define bins
        #magv, m_bins, cnt_bins = make_vc_bins(mag, mag_bins)

        # distance
        dist_bins = make_ibins(dist, ndist_bins)
        distv, _, _ = make_vc_bins(dist, dist_bins)

        # store data
        self.distv = distv

        # create array for conditional variables
        distv = self.fn_dist_scale(distv)
        #magv = self.fn_mag_scale(magv)
        # create conditional variables
        vc = distv
        #vc = np.concatenate( (distv, magv, ), axis=1)
        print('vc[:,0] min: ',vc[:,0].min())
        #print('vc[:,1] min: ',vc[:,1].min())


        # convert to tensors
        self.x_data = torch.from_numpy(x_data).float()
        # expand one dim
        self.x_data = self.x_data.unsqueeze(2)
        #self.vc = vc
        self.vc = torch.from_numpy(vc).float()
        print('self.x_data shape: ', self.x_data.shape)
        print('self.vc shape: ', self.vc.shape)

        # create vector with indices
        Nsamp = self.x_data.size(0)
        print('Nsamp', Nsamp)
        # index for variables
        self.ix = np.arange(Nsamp)

    def get_rand_cond_v(self,Nbatch):
        """
        :param Nbatch: number of samples to daraw
        :return:
        """
        ixc = np.random.choice(self.ix, size=Nbatch, replace=False)
        ixc.sort()
        vc_b =self.vc[ixc,:]
        return vc_b

    def __len__(self):
        return self.x_data.size(0)

    def __getitem__(self, idx):
        return ( self.x_data[idx,:], self.vc[idx, :] )
