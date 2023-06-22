# %%
import os
import json
import shutil
import h5py

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import datasets
import torchvision.transforms as transforms
from scipy import ndimage
import scipy.io

# DATA SET FOR POISSON 2D PROBLEM


# OUTPUT DIRECTORIES
DATA_DIR = '/scratch/mflorez/fourier/ffn1D/data/raw_burgers'
OUT_DIR = '//scratch/mflorez/fourier/ffn1D/data/pde1D'
FIG_DIR = '/home/mflorez/neural_ops/figs'
# target for down smapling factors
R = 2**3
# %%
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



# %%
# -------- Preparing data ---------
# load train data
file_train = os.path.join(DATA_DIR, 'burgers.mat')
reader = MatReader(file_train)
x_data = reader.read_field('a')
y_data = reader.read_field('u')
# load test data

print("------- Initial shapes --------")
print("x_train shape: ", x_data.shape)
print("y_train shape: ", y_data.shape)

# %%
# Downsample data
L0 = y_data.shape[1]
Lt = L0//R
print("Lt: ", Lt)

#%%
x_data = x_data[:, ::R]
y_data = y_data[:, ::R]

print("------- Downsampling results -----------")
print("y_train shape: ", y_data.shape)
print("y_train min: ", y_data.min())
print("y_train max: ", y_data.max())

# %%
# # make some plots
# fig_file = os.path.join(FIG_DIR, f"pde2D_{Lx}x{Lx}.png")
# stl = f"Solutions to Poison 2D {Lx}x{Lx}"
# make_plots_sol2D(x_test, y_test, Np=4, fig_file=fig_file, stitle=stl, show_fig=True, show_cbar=True)

#%% merge test and training sets
print("------- FINAL SHAPES --------")
print("x_data shape: ", x_data.shape)
print("y_data shape: ", y_data.shape)


# %%
# save  data all data
x_data_file = os.path.join(OUT_DIR, f"x_data")
y_data_file = os.path.join(OUT_DIR, f"y_data")
np.save(x_data_file, x_data)
np.save(y_data_file, y_data)

print('Done!')

# %%
