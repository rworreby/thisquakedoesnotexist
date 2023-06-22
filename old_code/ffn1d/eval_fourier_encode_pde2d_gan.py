# %%

import os
import json
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# ------------ USES POISSON 2D DATA SET ------
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
    'batch_size': 128,
    'epochs': 10000,
    # -------- FNO GAN configuration -----
    # Parameter for gaussian random field
    'alpha': 2.0,
    'tau': 3.0,
    # Size of input image to discriminator (lx,lx)
    'lx': 71,
    # padding before FFT
    'padding': 9,
    # Adam optimizser parasmeters:
    'lr': 1e-5,
    # -------- Generator params ----------------
    # number of fourier modes
    'modes': 12,
    # dimension of upsampling operator width
    'wd': 32,
    # gan file: architecture definitions
    'gan_file': 'gan_fno2d.py',
    # ------------------------- output configuration -------------------
    'out_dir': '/scratch/mflorez/fourier/gan2d_5pt/batch128_enc_lr5',
    'print_every': 400,
    'save_every': 1,
}

# ----------- Plotting configuration ------------
# number of synthetic plots to generate
NPLT_MAX = 5
# number of rows in plots
NPLT_ROWS = 5
# number of columns
NPLT_COLS = 1


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
Lx = config_d['lx']
# Padding
PADDING = config_d['padding']
# number of fourier modes
MODES = config_d['modes']
# dimension of upsampling operator width
WD = config_d['wd']

# %%
# flip data and labels
y_data = np.load(DATA_FILE)
x_data = np.load(ATTR_FILE)


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

# apply normalizers to dataset
x_normalizer = UnitGaussianNormalizer(x_data)
x_data = x_normalizer.encode(x_data)
y_normalizer = UnitGaussianNormalizer(y_data)
y_data = y_normalizer.encode(y_data)

# prepare data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_data, y_data), batch_size=NBATCH, shuffle=True)

# ----- test Loader-----
x_r, y_r = next(iter(train_loader))
print('shapes x_r, y_r: ', x_r.shape, y_r.shape)
Nb = x_r.size(0)

if Nb<NPLT_MAX:
    # batch size to small tile
    nn_samp = NPLT_MAX//Nb+1
    x_r = torch.tile(x_r,(nn_samp,1,1))
print('shapes tile x_r', x_r.shape)

#%%
fig_file = os.path.join(FIG_DIR, 'enc_pde2D.png')
stl = "Encoded Poisson 2D"
make_plot_imgs(x_r, fig_file, show_fig=SHOW_FIG, stitle=stl, Nb=NPLT_MAX, Nrows=NPLT_ROWS, Ncols=NPLT_COLS, fig_size=(2,8), show_cbar=True)
#%%
# -------- Test random field -------
# Test random field dim=2
grf = GaussianRF_odd(2, Lx, device=device)
z = grf.sample(NPLT_MAX)
print("z shape: ", z.shape)
z = z.detach().cpu()
# Make plot
fig_file = os.path.join(FIG_DIR, 'grf1.png')
stl = "Gaussian Random Field"
make_plot_imgs(z, fig_file, show_fig=SHOW_FIG, stitle=stl, Nb=NPLT_MAX, Nrows=NPLT_ROWS, Ncols=NPLT_COLS, fig_size=(2,8), show_cbar=True)


#%%
# ----- Testing models ------
import gan_fno2d
reload(gan_fno2d)
from gan_fno2d import Generator, Discriminator

# ----- test Discriminator-----
x_r, y_r = next(iter(train_loader))
# get discriminator
Nb = x_r.size(0)
x_r = x_r.to(device)
x_r = torch.unsqueeze(x_r,dim=-1)
print('shapes x_r: ', x_r.shape)
D = Discriminator(MODES, MODES, WD, lx=Lx, padding=PADDING).to(device)
print(D)
x_d = D(x_r)
print("D(x_b) shape: ", x_d.shape)
nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
print("Number discriminator parameters: ", nn_params)

#%%
# ----- test Generator ---------

G = Generator(MODES, MODES, WD,padding=PADDING).to(device)
print(G)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number Generator parameters: ",nn_params)

# Test model
# get GRF
z = grf.sample(NPLT_MAX)
z = torch.unsqueeze(z,dim=-1)
print("z shape: ", z.shape)
x_g = G(z)
print('G(z) shape: ', x_g.shape)
# remove extra dim
x_g = x_g.squeeze()
x_g = x_g.detach().cpu()
fig_file = os.path.join(FIG_DIR, 'syn_imgs.png')
stl = "Generator Encoded one pass"
make_plot_imgs(x_g, fig_file, show_fig=SHOW_FIG, stitle=stl, Nb=NPLT_MAX, Nrows=NPLT_ROWS, Ncols=NPLT_COLS, show_cbar=True)

#%%
# create new model instances
D = Discriminator(MODES, MODES, WD, lx=Lx, padding=PADDING).to(device)
G = Generator(MODES, MODES, WD, padding=PADDING).to(device)
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

y_normalizer.cuda()
x_normalizer.cuda()
for epoch in range(NUM_EPOCHS):
    # accumulator for losses
    d_train_loss = 0.0
    g_train_loss = 0.0

    for batch_i, (x_r, _) in enumerate(train_loader):
        # current batch size
        Nb = x_r.size(0)
        # -------- RESCALE IMAGES --------
        # rescale input images from [0,1) to [-1, 1)
        #x_r = 2.0*x_r - 1.0

        # --------  TRAIN THE DISCRIMINATOR --------
        d_optimizer.zero_grad()
        x_r = x_r.to(device)
        x_r = torch.unsqueeze(x_r, dim=-1)
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
        z = torch.unsqueeze(z,dim=-1)
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
        z = torch.unsqueeze(z,dim=-1)
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
    z = torch.unsqueeze(z, dim=-1)
    x_g = G(z)
    x_g = x_g.view(-1, Lx, Lx)
    # decode
    x_g = x_normalizer.decode(x_g)
    # move to CPU
    x_g = x_g.detach().cpu().numpy()
    fig_file = os.path.join(FIG_DIR, f"syn_ep_{epoch}.png")
    stl = f"Epoch={epoch}"
    make_plot_imgs(x_g, fig_file, stitle=stl, Nb=NPLT_MAX, Nrows=NPLT_ROWS, Ncols=NPLT_COLS, fig_size=(2,8), show_cbar=False, show_fig=False)
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
plt.title("Loss GAN fourier 71x71")
plt.savefig(fig_file, format='png')
#plt.show()
#%%




