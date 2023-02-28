#%%

import os
import json
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------ USES POISSON 2D DATA SET -------
"""
    Uses pointwise encoder using unit gaussian
    NO TANH activation at the end of generator model
   
"""

# ---------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imp import reload
# ------ optimizers ------
from Adam import Adam

# ------ optimizers ------

from torch.utils.data import DataLoader
from torch import nn

from utils import *
from random_fields import *

# -------- Input Parameters -------
config_d = {
    'data_file': '/scratch/mflorez/gmpw/train_ffn1D/train_D100/downsamp_5x_sel.npy',
    'attr_file': '/scratch/mflorez/gmpw/train_ffn1D/train_D100/vc_sel.npy',

    # -------- traning configuration ---------
    'batch_size': 40,
    'epochs': 400,
    # -------- FNO GAN configuration -----
    # Parameter for gaussian random field

    'alpha': 0.6,
    'tau': 10.0,
    # Size of input image to discriminator (lx,lx)
    'lt': 1000,
    # padding before FFT
    'padding': 9,
    # Adam optimizser parasmeters:
    'lr': 1e-4,
    # -------- Generator params ----------------
    # number of fourier modes
    'modes': 64,
    # dimension of upsampling operator width
    'wd': 128,
    # gan file: architecture definitions
    'gan_file': 'gan_fno1d.py',
    # ------------------------- output configuration -------------------
    #'out_dir': '/scratch/mflorez/fourier/ffn1D/exp/m64_lr5_2',
    'out_dir': '/scratch/mflorez/fourier/ffn1D/exp_disc/m64_lr4',
    'print_every': 400,
    'save_every': 1,
}

# ----------- Plotting configuration ------------
# number of synthetic plots to generate
NPLT_MAX = 20
# will display figures?
SHOW_FIG = True
# device to use
device = torch.device('cuda')

# setup folders to store experiments
def init_gan_conf(conf_d):
    """
    Output configurations dictionary and ddcreate data directory
    """
    out_dir = conf_d['out_dir']
    gan_file = conf_d['gan_file']

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = os.path.join(out_dir, 'config_d.json')
    # write out config file
    with open(fname, 'w') as f:
        json.dump(conf_d, f, indent=4)
    # create diretory for modelsaa
    models_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    # create directory for figures
    fig_dir = os.path.join(out_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # make a copy of gan.py which contains the model specifications
    gan_file = os.path.join('./', gan_file)
    gan_out = os.path.join(out_dir, gan_file)
    shutil.copyfile(gan_file, gan_out)
    return out_dir, models_dir, fig_dir


# Get paths
OUT_DIR, MODEL_DIR, FIG_DIR = init_gan_conf(config_d)
print(OUT_DIR, MODEL_DIR)
print(config_d)

# ----------------------- Read Params -------------------
# -> training hyperparams
DATA_FILE = config_d['data_file']
ATTR_FILE = config_d['attr_file']
# Batch size
NBATCH = config_d['batch_size']
# epochs for training
NUM_EPOCHS = config_d['epochs']
# how often to print samples
PRINT_EVERY = config_d['print_every']

# optimizer parameters default values
lr = config_d['lr']
# -------- Generator params ------
# Parameter for gaussian random field
ALPHA = config_d['alpha']
TAU = config_d['tau']
# Size of input image to discriminator (Lx,Lx)
Lt = config_d['lt']
# Padding
PADDING = config_d['padding']
# number of fourier modes
MODES = config_d['modes']
# dimension of upsampling operator width
WD = config_d['wd']


# %%
# flip data and labels
x_data = np.load(DATA_FILE)
y_data = np.load(ATTR_FILE)

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

print("y_data shape: ", y_data.shape)
print("y_data min: ", y_data.min())
print("y_data max: ", y_data.max())

# total number of training samples
Ntrain = y_data.shape[0]

# convert to torch tensors
x_data = torch.from_numpy(x_data).float()
y_data = torch.from_numpy(y_data).float()

# dataset preprocessing
x_data = x_data.unsqueeze(2)

# # apply normalizers to dataset
# x_normalizer = UnitGaussianNormalizer(x_data)
# x_data = x_normalizer.encode(x_data)
# y_normalizer = UnitGaussianNormalizer(y_data)
# y_data = y_normalizer.encode(y_data)

# prepare data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_data, y_data), batch_size=NBATCH, shuffle=True)

# ----- test Loader-----
x_r, y_r = next(iter(train_loader))
print('shapes x_r, y_r: ', x_r.shape, y_r.shape)
Nb = x_r.size(0)


#%%
DT = 0.05
fig_file = os.path.join(FIG_DIR, 'waves_real.png')
stl = "Real Accelerograms 1C"
x_r = x_r.squeeze()
plot_waves_1C(x_r, DT, t_max=50.0, show_fig=True, fig_file=fig_file,stitle=stl, color='C0')

#%%
# # Parameter for gaussian random field
# ALPHA = 2.0
# TAU = 3.0

# Test random field dim=1
# Parameter for gaussian random field

grf = GaussianRF_1C(1, Lt, alpha=ALPHA, tau=TAU, device=device)
z = grf.sample(NPLT_MAX)
print("z shape: ", z.shape)
z = z.squeeze()
z = z.detach().cpu()
# Make plot
fig_file = os.path.join(FIG_DIR, 'grf_1C.png')
stl = f"GRF 1D alpha={ALPHA}, tau={TAU}"
plot_waves_1C(z, DT, t_max=50.0, show_fig=True, fig_file=fig_file,stitle=stl, color='C4')

#%%
# ----- Testing models ------
import gan_fno1d
reload(gan_fno1d)
from gan_fno1d import Generator, Discriminator
# ----- test Discriminator-----
x_r, y_r = next(iter(train_loader))
# get discriminator
Nb = x_r.size(0)
x_r = x_r.to(device)
print('shapes x_r: ', x_r.shape)
D = Discriminator(MODES, WD,lt=Lt, padding=PADDING).to(device)
nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
print("Number discriminator parameters: ", nn_params)
print(D)
x_d = D(x_r)
print("D(x_b) shape: ", x_d.shape)

#%%
# define small kernel neural network
kernelNet = nn.Sequential(
                nn.Linear(1, 16), torch.nn.GELU(),
                nn.Linear(16, 1),
            ).to(device)
x = torch.ones(4,10,dtype=torch.float32,device=device)
x[1,:] = 2.0*x[1,:]
x[2,:] = 3.0*x[2,:]
x[3,:] = 4.0*x[3,:]
x = x.unsqueeze(-1)
print(x.shape)
print(kernelNet)
y = kernelNet(x)
print(y.shape)

yg = D.get_grid(x.shape,x.device)

# Calculate inner product
k = torch.einsum('bik,bik->bk', x, yg)
print(k.shape)
print(k)

#%%