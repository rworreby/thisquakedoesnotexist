#%%

import os
import json
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ------------ USES POISSON 2D DATA SET ----------
"""
    Uses pointwise encoder using unit gaussian
    NO TANH activation at the end of generator model
    W GAN implementation
   
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
from torch.autograd import Variable
from torch import nn

from utils import *
from random_fields import *


# -------- Input Parameters -------
config_d = {
    'data_file': '/scratch/mflorez/gmpw/train_ffn1D/train_D100/downsamp_5x_sel.npy',
    'attr_file': '/scratch/mflorez/gmpw/train_ffn1D/train_D100/vc_sel.npy',

    # -------- traning configuration ---------
    'batch_size': 40,
    'epochs': 4,
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
        # ---- wassertddain gan configuration -----
    # number of iterations for critic
    'n_critic': 5,
    'gp_lambda': 10.0,
    # Adam optimizser parasmeters :
    'beta1': 0.0,
    'beta2': 0.9,

    # -------- Generator params ----------------
    # number of fourier modes
    'modes': 64,
    # dimension of upsampling operator width
    'wd': 128,
    # gan file: architecture definitions
    'gan_file': 'gan_fno1d.py',
    # ------------------------- output configuration -------------------
    #'out_dir': '/scratch/mflorez/fourier/ffn1D/exp/m64_lr5_2',
    'out_dir': '/scratch/mflorez/fourier/ffn1D/exp_wgan/m64_lr4_nobeta',
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

# WGAN
# iterations for critic
N_CRITIC = config_d['n_critic']
# optimizer parameters default values
BETA1 = config_d['beta1']
BETA2 = config_d['beta2']
GP_LAMBDA = config_d['gp_lambda']


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
# generate samples
x_r, y_r = next(iter(train_loader))
# get discriminator
Nb = x_r.size(0)
x_r = x_r.to(device)
z = grf.sample(Nb)
x_g = G(z)

# # ----- test gradient penalty -----
#
# gradient penalty
Nsamp = x_r.size(0)
# random alpha
alpha = torch.rand(Nsamp, 1, 1, device=device)
# Get random interpolation between real and fake samples
# Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
Xp = (alpha * x_r + ((1 - alpha) * x_g)).requires_grad_(True)
# apply dicriminator
D_xp = D(Xp)
#Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
Xout = Variable(torch.ones(Nsamp,1, device=device), requires_grad=False)
# Get gradient w.r.t. interpolates
grads = torch.autograd.grad(
    outputs=D_xp,
    inputs=Xp,
    grad_outputs=Xout,
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
)[0]
grads = grads.view(grads.size(0), -1)
regularizer = ((grads.norm(2, dim=1) - 1) ** 2).mean()


print('grads.shape', grads.shape)
print('regularizer.shape', regularizer.shape)


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
d_wloss_ep = np.zeros(NUM_EPOCHS)
d_total_loss_ep = np.zeros(NUM_EPOCHS)
g_loss_ep = np.zeros(NUM_EPOCHS)


for epoch in range(NUM_EPOCHS):
    # store train losses
    d_train_wloss = 0.0
    d_train_gploss = 0.0
    g_train_loss = 0.0
    # counter for critic iterations
    n_critic = 0
    for batch_i, (x_r, _) in enumerate(train_loader):
        # current batch size
        Nb = x_r.size(0)
        # --------  TRAIN THE DISCRIMINATOR --------
        # 1. Move data to GPU
        x_r = x_r.to(device)
        # clear gradients
        d_optimizer.zero_grad()
        # 2. Get fake data
        # Gaussian ranfom field
        z = grf.sample(Nb)
        x_g = G(z)
        # 3. Compute gradient penalty term
        # ---------- GRADIENT PENALTY
        # random alpha
        alpha = torch.rand(Nb, 1, 1, device=device)
        # Get random interpolation between real and fake samples
        Xp = (alpha * x_r + ((1 - alpha) * x_g)).requires_grad_(True)
        # apply dicriminator
        D_xp = D(Xp)
        # Jacobian related variable
        Xout = Variable(torch.ones(Nb,1, device=device), requires_grad=False)
        # Get gradient w.r.t. interpolates
        grads = torch.autograd.grad(
            outputs=D_xp,
            inputs=Xp,
            grad_outputs=Xout,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.view(grads.size(0), -1)
        # 4. Compute losses
        d_gp_loss = GP_LAMBDA*((grads.norm(2, dim=1) - 1) ** 2).mean()
        d_w_loss = -torch.mean(D(x_r)) + torch.mean(D(x_g))
        d_loss = d_w_loss + d_gp_loss

        # store losses in accumulators
        d_train_wloss += d_w_loss.item()
        d_train_gploss += d_gp_loss.item()

        # 5. Calculate gradients
        d_loss.backward()
        # 6. update model weights -> run optimizer
        d_optimizer.step()

        # add iteration for critic
        n_critic += 1


        # --------------  TRAIN THE GENERATOR ------------------------
        # make sure discriminator has performed desired number of iterations
        if n_critic == N_CRITIC:
            # take a generator step every n_critic generator iterations
            # set initial gradients to zero
            g_optimizer.zero_grad()

            # 1. Generate fake data with fake data
            z = grf.sample(Nb)
            x_g = G(z)
            # 2. calculate loss
            g_loss = -torch.mean( D(x_g) )
            # The wights of the generator are optimized with respect
            # to the discriminator loss
            # the generator is trained to produce data that will be classified
            # by the discriminator as "real"

            # Calculate gradients for generator
            g_loss.backward()
            # update weights for generator -> run optimizer
            g_optimizer.step()
            # store loss
            # multiply by NCRITIC for CONSISTENCY
            g_train_loss += N_CRITIC*g_loss.item()

            # resete critic iterations to zero
            n_critic = 0

            ### --------------  END GENERATOR STEP ------------------------

        # Print some loss stats
        if (batch_i % PRINT_EVERY==0):
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_tot_loss: {:6.4f} | g_wloss: {:6.4f}'.format(
                epoch + 1, NUM_EPOCHS, d_loss.item(), d_w_loss.item()))

    # --------- End training epoch -----------
    # save losses
    d_wloss_ep = d_train_wloss/Ntrain
    d_total_loss_ep = (d_train_wloss+d_train_gploss)/Ntrain
    g_loss_ep = g_train_loss/Ntrain

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
    # back to train mode
    G.train()



#%%
# make lot of losses
iep = np.arange(NUM_EPOCHS) + 1.0
fig_file = os.path.join(OUT_DIR, f"gan_losses.png")
plt.figure(figsize= (8,6))
plt.plot(iep, d_wloss_ep, color='C0',label="W Distance")
plt.plot(iep, d_total_loss_ep, color='C1',label="D Loss")
plt.plot(iep, g_loss_ep, color='C2', label="G Loss")
plt.legend()
plt.ylabel('W Losses')
plt.xlabel('Epoch')
plt.title("Wasserstein GAN 1D  Fourier, 1C")
plt.savefig(fig_file, format='png')
plt.show()
#%%




