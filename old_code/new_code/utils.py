import torch
import numpy as np
# import scipy.io
# import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------------------------  helful functions------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_waves_3C(ws, dt, t_max=50.0, fig_file=None, Np=10, fig_size = (8,6), stitle=None, show_fig=False, color='C0'):
    """
        Make plots of 3C waves
        shape of the data has the form -> (Nobs, 3, Nt)
        :param ws_p: numpy.ndarray of shape (Nobs, 3, Nt)
        :param Np: number of 3C waves to show
        :param color: Color of the plot
        :param t_max: max time for the plot in sec
        :return: show a panel with the waves

        """
    Nt = int(t_max / dt)
    if ws.shape[0] < Np:
        # in case there are less observations than points to plot
        Np = ws.shape[0]


    ws_p = ws[:Np, :, :Nt]

    # select waves to plot
    tt = dt * np.arange(Nt)
    plt.figure()
    fig, axs = plt.subplots(Np, 3, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                            figsize=fig_size)
    for i in range(Np):
        for j in range(3):
            wf = ws_p[i, j, :]
            if Np > 1:
                axs[i, j].plot(tt, wf, color=color)
            else:
                axs[j].plot(tt, wf, color=color)

    for ax in axs.flat:
        ax.label_outer()

    if stitle is not None:
        fig.suptitle(stitle,fontsize=12)
    if show_fig:
        fig.show()
    if fig_file is not None:
        fig.savefig(fig_file, format='png')
        if not show_fig:
            plt.close()
    else:
        fig.show()

def plot_waves_1C(ws, dt, t_max=50.0, ylim=None, fig_file=None, Np=12, fig_size = (6,9), stitle=None, show_fig=False, color='C0'):
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
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                            figsize=fig_size)
    for ik in range(Np):
        wf = ws_p[ik,:]
        axs[ik].plot(tt, wf, color=color)
        if ylim is not None:
            axs[ik].set_ylim(ylim)


    for ax in axs.flat:
        ax.label_outer()

    if stitle is not None:
        fig.suptitle(stitle,fontsize=12)

    if show_fig:
        fig.show()
    if fig_file is not None:
        fformat = fig_file.split('.')[-1]
        print('saving:',fformat)
        fig.savefig(fig_file, format=fformat)
        if not show_fig:
            plt.close()
    else:
        fig.show()


# ploting images
def make_plot_imgs(x_p, fig_file=None, Nb=64, Nrows = 6, Ncols = 8, fig_size = (8,6) , colormap='hot', stitle=None, show_fig=False, show_cbar=True):
    """
    Make panels with images

    :param x_p: array with data (Nb, Lx, Ly)
    :param fig_file: full path
    :param Nb: Batch size
    :param Nrows: N rows to show
    :param Ncols: N cols to show
    :param fig_size: Size of figure
    :param stitle: Title
    :param show_fig:  call fig.show()
    :param show_cbar:  add color bar
    :return: saves the figure
    """
    nn_ps = Nrows*Ncols
    assert nn_ps<=Nb, "Nrows*Ncols must be smaller than the batch_size"
    # make figure no space within images
    fig, axs = plt.subplots(
            nrows=Nrows, ncols=Ncols,
            sharex='col', sharey='row',
            subplot_kw={'xticks': [], 'yticks': []},
            gridspec_kw={'hspace': 0.0, 'wspace': 0.0},
            figsize=fig_size,
            #constrained_layout=True
    )
   # number of plots
    nn_ps = Nrows*Ncols
    ixs = np.random.choice(Nb,nn_ps,replace=False).reshape(Nrows,Ncols)
    # Display the images
    for col in range(Ncols):
        for row in range(Nrows):
            if Ncols > 1:
                ax = axs[row, col]
            else:
                ax = axs[row]
            pcm = ax.imshow(x_p[ixs[row,col],:,:],cmap=colormap)
            #fig.colorbar(pcm, ax=ax)
    # remove outer labels
    for ax in axs.flat:
        ax.label_outer()

    if show_cbar:
        if Ncols > 1:
            fig.colorbar(pcm, ax=axs[:, :], shrink=0.6, location='right')
        else:
            fig.colorbar(pcm, ax=axs[:], shrink=0.6, location='right')

    if stitle is not None:
        fig.suptitle(stitle,fontsize=12)
    if show_fig:
        fig.show()
    if fig_file is not None:
        fig.savefig(fig_file, format='png')
        if not show_fig:
            plt.close()
    else:
        fig.show()

# ----- PLOT solutions to 2D PDE -----
def make_plots_sol2D(cp, up, Np=4, fig_file=None, fig_size=(6, 12), colormap='hot', stitle=None, show_fig=False, show_cbar=True):
    """
    Make panels with images
    Left is the Coeff(x,y)
    Right is the function u(x,y)

    :param cp: array with Coeff(x,y) shape (?, Lx, Ly)
    :param up: up array with solutions u(x,y) shape (?, Lx, Ly)
    :param Np: Number of points to sample
    :param fig_file: full path
    :param stitle: Title
    :param show_fig:  call fig.show()
    :param show_cbar:  add color bar
    :return: saves the figure
    """
    Nb = up.shape[0]
    assert Np <= Nb, "Nb must be smaller than numper of samples Nb"
    # make figure no space within images
    Nrows = 2
    fig, axs = plt.subplots(
        nrows=Np, ncols=Nrows,
        sharex='col', sharey='row',
        subplot_kw={'xticks': [], 'yticks': []},
        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},
        figsize=fig_size,
        # constrained_layout=True
    )

    # get random sample to plot
    ixs = np.random.choice(Nb, Np, replace=False)
    # Display the images
    for ik in range(Np):
        ax = axs[ik,0]
        ax.imshow(cp[ixs[ik], :, :], cmap=colormap)
        ax = axs[ik,1]
        pcm = ax.imshow(up[ixs[ik], :, :], cmap=colormap)
    # remove outer labels
    for ax in axs.flat:
        ax.label_outer()
    if show_cbar:
        fig.colorbar(pcm, ax=axs[:, :], shrink=0.6, location='right')
    if stitle is not None:
        fig.suptitle(stitle, fontsize=12)
    if show_fig:
        fig.show()
    if fig_file is not None:
        fig.savefig(fig_file, format='png')
        if not show_fig:
            plt.close()
    else:
        fig.show()


# ------ Load Matlab data ----------
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# ------ normalization, pointwise gaussian -----------
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# Pointwise normalization -1.0,1.0
class Normalizer_11(object):
    def __init__(self, x):
        super(Normalizer_11, self).__init__()
        # store min and max
        self.v_min = torch.min(x, 0)[0]
        self.v_max = torch.max(x, 0)[0]
        # range tensor
        self.dv = self.v_max-self.v_min

    def encode(self, x):
        xn = (x-self.v_min) / self.dv
        # rescale to [-1,1]
        xn = 2.0*xn -1.0
        return xn

    def decode(self, x):
        # back to the real scale
        xr = (x+1.0)/2.0
        xr = xr*self.dv+self.v_min
        return xr

    def cuda(self):
        # move variables to cuda
        self.v_min = self.v_min.cuda()
        self.v_max = self.v_max.cuda()
        self.dv = self.dv.cuda()

    def cpu(self):
        # move variables to cpu
        self.v_min = self.v_min.cpu()
        self.v_max = self.v_max.cpu()
        self.dv = self.dv.cpu()



# Loss function
# -------------- loss function with rel/abs Lp loss ---------
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)



