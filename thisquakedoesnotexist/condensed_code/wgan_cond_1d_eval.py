#%%

import os
import json
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------ USES POISSON 2D DATA SET ---------------------
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
import torch.optim as optim

# ------ optimizers ------

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn

from torch.utils.data import DataLoader
from torch import nn

# ---- custom utilities ---
from thisquakedoesnotexist.utils.data_utils import WaveDatasetDist
from thisquakedoesnotexist.utils.random_fields import *
from thisquakedoesnotexist.utils.utils import *
# from thisquakedoesnotexist.plotting.fn_plot import *

# -------- Input Parameters -------
config_d = {
    'data_file': '../data/japan/waveforms.npy',
    # 'data_file': '../data/japan/downsampled.h5',
    'attr_file': '../data/japan/attributes.csv',

    # -------- traning configuration -----------
    'batch_size': 40,
    'epochs': 1000,
    # ---------- GAN configuration ----
    # Size of input image to discriminator (lx,)
    'lt': 1000,
    # dimension of Gaussian noise
    'noise_dim': 100,
    # Adam optimizser parasmeters:
    'lr': 1e-4,
    # ---- wassertddain gan configuration -------
    # number of iterations for critic
    'n_critic': 5,
    'gp_lambda': 10.0,
    # Adam optimizser parasmeters :
    'beta1': 0.0,
    'beta2': 0.9,
    # gan file: architecture definitions
    'gan_file': 'gan1d.py',
    # Conditional Configuration
    # number of distance bins
    'ndist_bins': 10,
    # ---------------------------- output configuration ---------------------
    #'out_dir': '/scratch/mflorez/fourier/ffn1D/exp/m64_lr5_2',
    'out_dir': '../data/output',
    'print_every': 400,
    'save_every': 1,
}

# ----------- Plotting configuration ------------
# number of synthetic plots to generate
NPLT_MAX = 20

# Bins for validation
DIST_DICT= {
    (0,0): [40.00, 54.01],
    (0,1): [54.01, 68.03],
    (0,2): [68.03, 82.04],
    (1,0): [82.04, 96.05],
    (1,1): [96.05, 110.06],
    (1,2): [110.06, 124.07],
}


# will display figures?
SHOW_FIG = False
# device to use
device = torch.device('cuda')

# setup folders to store experiments
def init_gan_conf(conf_d):
    """
    Output configurations dictionary and create data directory
    """
    out_dir = conf_d['out_dir']
    gan_file = conf_d['gan_file']

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = os.path.join(out_dir, 'config_d.json')
    # write out config file
    with open(fname, 'w') as f:
        json.dump(conf_d, f, indent=4)
    # create diretory for models
    models_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    # create directory for figures
    fig_dir = os.path.join(out_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # create dir for stats
    tt_stats_dir = os.path.join(out_dir, 'time_stats')
    if not os.path.exists(tt_stats_dir):
        os.makedirs(tt_stats_dir)

    # make a copy of gan.py which contains the model specifications
    gan_file = os.path.join('./', gan_file)
    gan_out = os.path.join(out_dir, gan_file)
    shutil.copyfile(gan_file, gan_out)
    return out_dir, models_dir, fig_dir, tt_stats_dir


# Get paths
OUT_DIR, MODEL_DIR, FIG_DIR, TT_STATS_DIR = init_gan_conf(config_d)
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
BETA1 = config_d['beta1']
BETA2 = config_d['beta2']

# -------- Generator params ------
# Size of input image to discriminator (Lx,)
Lt = config_d['lt']
NOISE_DIM = config_d['noise_dim']

# WGAN
# iterations for critic
N_CRITIC = config_d['n_critic']
# optimizer parameters default values
GP_LAMBDA = config_d['gp_lambda']
NDIST_BINS = config_d['ndist_bins']
# Dataset sampling intervals
DT = 0.04
T_MAX = DT*Lt



# %%


# # apply normalizers to dataset
# x_normalizer = UnitGaussianNormalizer(x_data)
# x_data = x_normalizer.encode(x_data)
# y_normalizer = UnitGaussianNormalizer(y_data)
# y_data = y_normalizer.encode(y_data)
# prepare data loader
dataset = WaveDatasetDist(data_file=DATA_FILE, attr_file=ATTR_FILE, ndist_bins=NDIST_BINS, dt=DT )
# create train loader
train_loader = DataLoader(dataset, batch_size=NBATCH, shuffle=True)

Ntrain = len(dataset)


#%%
# ----- test Loader-----
x_r, y_r = next(iter(train_loader))
print('shapes x_r, y_r: ', x_r.shape, y_r.shape)
Nb = x_r.size(0)


fig_file = os.path.join(FIG_DIR, 'waves_real.png')
stl = "Real Accelerograms 1C"
x_r = x_r.squeeze()
plot_waves_1C(x_r, DT, t_max=T_MAX, show_fig=True, fig_file=fig_file,stitle=stl, color='C0')

#%%

# Test random field dim=1
# Parameter for gaussian random field

grf = rand_noise(1, NOISE_DIM, device=device)
z = grf.sample(NPLT_MAX)
print("z shape: ", z.shape)
z = z.squeeze()
z = z.detach().cpu()
# Make plot
fig_file = os.path.join(FIG_DIR, 'rand_1C.png')
stl = f"Random Gaussian Noise"
plot_waves_1C(z, DT, t_max=4.0, show_fig=True, fig_file=fig_file,stitle=stl, color='C4')


#%%
# ----- Testing models ------
import gan1d
reload(gan1d)
from gan1d import Generator, Discriminator
# ----- test Discriminator-----
x_r, vc_r = next(iter(train_loader))
# get discriminator
Nb = x_r.size(0)
x_r = x_r.to(device)
vc_r = vc_r.to(device)
print('shapes x_r: ', x_r.shape)
D = Discriminator().to(device)
nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
print("Number discriminator parameters: ", nn_params)
print(D)
x_d = D(x_r, vc_r)
print("D(x_r, vc_r) shape: ", x_d.shape)

#%%
#----- Test generator -----------
NPLT_MAX = Nb
G = Generator(z_size=NOISE_DIM).to(device)
print(G)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number Generator parameters: ", nn_params)
z = grf.sample(NPLT_MAX)
print("z shape: ", z.shape)
# get random batch of conditional variables
vc_g = dataset.get_rand_cond_v(NPLT_MAX)
vc_g = vc_g.to(device)
x_g = G(z, vc_g)
print('G(z) shape: ', x_g.shape)
x_g = x_g.squeeze(1)
# make plot
fig_file = os.path.join(FIG_DIR, 'syn_test_onepass.png')
stl = "Generator one pass"
x_g = x_g.squeeze()
x_g = x_g.detach().cpu()
plot_waves_1C(x_g, DT, t_max=T_MAX, show_fig=True, fig_file=fig_file,stitle=stl, color='C1')


#%%
x_r, vc_r = next(iter(train_loader))
# get discriminator
Nb = x_r.size(0)
x_r = x_r.to(device)
vc_r = vc_r.to(device)

z = grf.sample(Nb)
x_g = G(z,vc_r)

# # ----- test gradient penalty -----
#
# random alpha
alpha = torch.rand(Nb, 1, 1, device=device)
# Get random interpolation between real and fake samples
# Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
Xp = (alpha * x_r + ((1 - alpha) * x_g)).requires_grad_(True)
# apply dicriminator
D_xp = D(Xp,vc_r)
#Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
Xout = Variable(torch.ones(Nb, 1, device=device), requires_grad=False)
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

D = Discriminator().to(device)
G = Generator(z_size=NOISE_DIM).to(device)


d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=[BETA1,BETA2])
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=[BETA1,BETA2])



#%%
# train the network
D.train()
G.train()
d_wloss_ep = np.zeros(NUM_EPOCHS)
d_total_loss_ep = np.zeros(NUM_EPOCHS)
g_loss_ep = np.zeros(NUM_EPOCHS)


# distnace array
distv = 60.0*np.ones((NPLT_MAX,1))
distv = dataset.fn_dist_scale(distv)
distv = torch.from_numpy(distv).float()
distv = distv.to(device)

for ix_ep in range(NUM_EPOCHS):
    # store train losses
    d_train_wloss = 0.0
    d_train_gploss = 0.0
    g_train_loss = 0.0
    # counter for critic iterations
    n_critic = 0
    for batch_i, (x_r, vc_r) in enumerate(train_loader):
        # current batch size
        Nb = x_r.size(0)
        # --------  TRAIN THE DISCRIMINATOR --------
        # 1. Move data to GPU
        x_r = x_r.to(device)
        vc_r = vc_r.to(device)

        # clear gradients
        d_optimizer.zero_grad()
        # 2. Get fake data
        # Gaussian ranfom field
        z = grf.sample(Nb)
        x_g = G(z, vc_r)
        # 3. Compute gradient penalty term
        # ---------- GRADIENT PENALTY
        # random alpha
        alpha = torch.rand(Nb, 1, 1, device=device)
        # Get random interpolation between real and fake samples
        Xp = (alpha * x_r + ((1 - alpha) * x_g)).requires_grad_(True)
        # apply dicriminator
        D_xp = D(Xp, vc_r)
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
        d_w_loss = -torch.mean(D(x_r, vc_r)) + torch.mean(D(x_g, vc_r))
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
            vc_g = dataset.get_rand_cond_v(Nb)
            vc_g = vc_g.to(device)

            x_g = G(z, vc_g)
            # 2. calculate loss
            g_loss = -torch.mean( D(x_g, vc_g) )
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
                ix_ep + 1, NUM_EPOCHS, d_loss.item(), d_w_loss.item()))

    # --------- End training epoch -----------
    # save losses
    d_wloss_ep[ix_ep] = d_train_wloss/Ntrain
    d_total_loss_ep[ix_ep] = (d_train_wloss+d_train_gploss)/Ntrain
    g_loss_ep[ix_ep] = g_train_loss/Ntrain

    # --- AFTER EACH EPOCH ---
    # generate and save sample synthetic images
    # eval mode for generating samples
    G.eval()
    z = grf.sample(NPLT_MAX)
    x_g = G(z,distv)
    x_g = x_g.squeeze()
    x_g = x_g.detach().cpu()
    fig_file = os.path.join(FIG_DIR, f"syn_ep_{ix_ep}.png")
    stl = f"Epoch={ix_ep}, dist = 60 km"
    plot_waves_1C(x_g, DT, t_max=T_MAX, show_fig=False, fig_file=fig_file,stitle=stl, color='C1')
    # store model
    fmodel = os.path.join(MODEL_DIR, 'model_G_epoch_'+str(ix_ep)+'.pth')
    torch.save({'state_dict': G.state_dict()}, fmodel)

    # plot validation statistics
    fig_file = os.path.join(TT_STATS_DIR, f"val_plots_{ix_ep}.png")
    
    # TODO Robin: Fix plotting
    # make_val_figs_all(G, grf, dataset, DIST_DICT, fig_file, figsize=(9, 36), Ni=2, Nj=3, iD1=0, iD2=1, fontsize=9, device=device)

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
#plt.show()
#%%




