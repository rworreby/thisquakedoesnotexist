#%%

import os
import json
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    'data_file': '/scratch/mflorez/gmpw/train_ffn1D/train_D200/downsamp_5x_sel.npy',
    'attr_file': '/scratch/mflorez/gmpw/train_ffn1D/train_D200/wforms_table_sel.csv',

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
    'out_dir': '/scratch/mflorez/fourier/ffn1D/cond/m64_lr4',
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

#%%

from dataUtils import WaveDatasetDist

dataset = WaveDatasetDist(data_file=DATA_FILE, attr_file=ATTR_FILE, ndist_bins=4 )


# create train loader
train_loader = DataLoader(dataset, batch_size=NBATCH, shuffle=True)

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

x = np.linspace(0.0,1.0, num=1000, endpoint=False)
sigma= 0.05
C = 1/((sigma)*np.sqrt(2*np.pi))
gx = C*np.exp(-0.5*np.square(x-0.8)/(sigma**2))

plt.plot(x,gx)
plt.show()


#%%

#MAG_BINS = [5.8, 6.0, 6.2, 6.501]

# v = y_data[:,0]
# print(v.min())
# print(v.max())
#

# vc, m_bins, cnt_bins = make_vc_bins(v,MAG_BINS)
plt.hist(dist, bins=4 )
plt.show()

plt.hist(distb, bins=4 )
plt.show()

plt.hist(vc, bins=4 )
plt.show()

#%%
# convert to torch tensors
x_data = torch.from_numpy(x_data).float()
y_data = torch.from_numpy(y_data).float()

# dataset preprocessing
x_data = x_data.unsqueeze(2)


#%%


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
#----- Test generator -----------
G = Generator(MODES, WD, padding=PADDING).to(device)
print(G)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number Generator parameters: ", nn_params)
z = grf.sample(NPLT_MAX)
print("z shape: ", z.shape)
x_g = G(z)
print('G(z) shape: ', x_g.shape)
# make plot
fig_file = os.path.join(FIG_DIR, 'syn_test_onepass.png')
stl = "Generator one pass"
x_g = x_g.squeeze()
x_g = x_g.detach().cpu()
plot_waves_1C(x_g, DT, t_max=50.0, ylim=(-0.05,0.05), show_fig=True, fig_file=fig_file,stitle=stl, color='C1')


#%%
# create new model instances
D = Discriminator(MODES, WD,lt=Lt, padding=PADDING).to(device)
G = Generator(MODES, WD, padding=PADDING).to(device)

# Create optimizers for the discriminator and generator
g_optimizer = Adam(G.parameters(), lr=lr, weight_decay=1e-4)
d_optimizer = Adam(D.parameters(), lr=lr, weight_decay=1e-4)
fn_loss = nn.BCEWithLogitsLoss()

#%%
# train the network
D.train()
G.train()
d_loss_ep = np.zeros(NUM_EPOCHS)
g_loss_ep = np.zeros(NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    # accumulator for losses
    d_train_loss = 0.0
    g_train_loss = 0.0
    for batch_i, (x_r, _) in enumerate(train_loader):
        # current batch size
        Nb = x_r.size(0)

        # --------  TRAIN THE DISCRIMINATOR --------
        d_optimizer.zero_grad()
        x_r = x_r.to(device)
        # 1. Train with real images
        # Compute the discriminator losses on real data
        D_real = D(x_r)
        labels_ones = torch.ones_like(D_real)
        # calculate numerically stable loss
        d_real_loss = fn_loss(D_real, labels_ones)

        # 2. Train with fake images
        # Generate fake images
        # gaussian random numbers
        z = grf.sample(Nb)
        x_g = G(z)
        D_fake = D(x_g)
        # Discriminator loss using fake images
        labels_zeros = torch.zeros_like(D_fake)
        d_fake_loss = fn_loss(D_fake, labels_zeros)

        # Compute final loss
        d_loss = d_real_loss + d_fake_loss
        # store loss
        d_train_loss += d_loss.item()
        # Calculate gradients
        d_loss.backward()
        # Update wights -> run optimizer
        # backprojection step
        d_optimizer.step()

        # --------------  TRAIN THE GENERATOR ------------------------
        # set initial gradients to zero
        g_optimizer.zero_grad()

        # 1. Train with fake images and flipped labels
        # Generate fake images
        z = grf.sample(Nb)
        x_g = G(z)
        # forward step 2 -> pass through discriminator
        D_xg = D(x_g)

        # calculate the generator loss
        # labels = 1
        g_labels = torch.ones_like(D_xg)
        # calculate numerically stable loss
        g_loss = fn_loss(D_xg, g_labels)
        # store loss
        g_train_loss += g_loss.item()

        # The wights of the generator are optimized with respect
        # to the discriminator loss
        # the generator is trained to produce data that will be classified
        # by the discriminator as 1 "real"

        # Calculate gradients for generator
        g_loss.backward()
        # update weights for generator
        g_optimizer.step()

        # Print some loss stats
        if batch_i % PRINT_EVERY == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, NUM_EPOCHS, d_loss.item(), g_loss.item()))

        # ----- End training epoch ------
    # save loss
    d_loss_ep[epoch] = d_train_loss/Ntrain
    g_loss_ep[epoch] = g_train_loss/Ntrain
    # --- AFTER EACH EPOCH ---
    # generate and save sample synthetic images
    # eval mode for generating samples
    G.eval()
    z = grf.sample(NPLT_MAX)
    x_g = G(z)
    x_g = x_g.squeeze()
    x_g = x_g.detach().cpu()
    fig_file = os.path.join(FIG_DIR, f"syn_ep_{epoch}.png")
    stl = f"Epoch={epoch}"
    plot_waves_1C(x_g, DT, t_max=50.0, show_fig=False, fig_file=fig_file,stitle=stl, color='C1')
    # store model
    fmodel = os.path.join(MODEL_DIR, 'model_G_epoch_'+str(epoch)+'.pth')
    torch.save({'state_dict': G.state_dict()}, fmodel)

    # back to train mode
    G.train()

#%%
# make lot of losses
iep = np.arange(NUM_EPOCHS) + 1.0
fig_file = os.path.join(OUT_DIR, f"gan_losses.png")
plt.figure(figsize= (8,6))
plt.plot(iep, d_loss_ep, color='C0',label="Discriminator")
plt.plot(iep, g_loss_ep, color='C1', label="Generator")
plt.legend()
plt.ylabel('BCE Loss')
plt.xlabel('Epoch')
plt.title("JS Losses 1D Fourier, 1C")
plt.savefig(fig_file, format='png')
#plt.show()
#%%



