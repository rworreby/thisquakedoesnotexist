# %%

import os
import json
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
#from random_fields import *

# -------- Input Parameters -------
config_d = {
    'data_dir': '/scratch/mflorez/fourier/ffn1D/data/pde1D',
    # -------- training configuration ---------
    'batch_size': 20,
    'epochs': 500,
    # training points
    'ntrain': 1000,
    # testing points
    'ntest': 100,
    # ----- Optimizer ---------
    # Learning rate:
    'lr': 1e-3,
    # how to change learning rate
    'step_size': 100,
    'gamma': 0.5,
    # -------- FNO configuration -----
    # Size of input image to discriminator (lx,lx)
    'lx': 71,
    # padding before FFT
    'padding': 9,
    # number of fourier modes
    'modes': 12,
    # dimension of upsampling operator width
    'wd': 32,
    # gan file: architecture definitions
    'gan_file': 'gan_fno2d.py',
    # ------------------------- output configuration -------------------
    'out_dir': '/scratch/mflorez/fourier/solv2d/b20_lr3',
    'print_every': 400,
    'save_every': 1,
}
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
DATA_DIR = config_d['data_dir']
# Batch size
NBATCH = config_d['batch_size']
# epochs for training
NUM_EPOCHS = config_d['epochs']
NTRAIN = config_d['ntrain']
NTEST = config_d['ntest']
# scheduler
GAMMA = config_d['gamma']
STEP_SIZE = config_d['step_size']

# how often to print samples
PRINT_EVERY = config_d['print_every']

# optimizer parameters default values
lr = config_d['lr']
# -------- Generator params ------
# Size of input image to discriminator (Lx,Lx)
Lx = config_d['lx']
# Padding
PADDING = config_d['padding']
# number of fourier modes
MODES = config_d['modes']
# dimension of upsampling operator width
WD = config_d['wd']
# %%

# load train data
x_train_file = os.path.join(DATA_DIR, f"x_train.npy")
y_train_file = os.path.join(DATA_DIR, f"y_train.npy")
x_train = np.load(x_train_file, )
y_train = np.load(y_train_file, )
# get desired number of samples
x_train = x_train[:NTRAIN, :, :]
y_train = y_train[:NTRAIN, :, :]

# save test data
x_test_file = os.path.join(DATA_DIR, f"x_test.npy")
y_test_file = os.path.join(DATA_DIR, f"y_test.npy")
x_test = np.load(x_test_file, )
y_test = np.load(y_test_file, )
x_test = x_test[:NTEST, :, :]
y_test = y_test[:NTEST, :, :]

print("y_train shape: ", y_train.shape)
print("y_train min: ", y_train.min())
print("y_train max: ", y_train.max())
print("--------------------------")
print("y_test shape: ", y_test.shape)
print("y_test min: ", y_test.min())
print("y_test max: ", y_test.max())

# make some plots
fig_file = os.path.join(OUT_DIR, f"pde2D_sol_{Lx}x{Lx}.png")
stl = f"Solutions to Poison 2D training data {Lx}x{Lx}"
make_plots_sol2D(x_train, y_train, Np=4, fig_file=fig_file, stitle=stl, show_fig=True, show_cbar=True)

# Load into torch tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

# %%
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)

print("y_train encoded shape: ", y_train.shape)
print("y_train encoded min: ", y_train.min())
print("y_train encoded max: ", y_train.max())


fig_file = os.path.join(OUT_DIR, f"pde2D_enc_sol_{Lx}x{Lx}.png")
stl = f"Encoded Sol to Poison 2D training data {Lx}x{Lx}"
make_plots_sol2D(x_train, y_train, Np=4, fig_file=fig_file, stitle=stl, show_fig=True, show_cbar=True)

#%%
# reshape inputs
x_train = x_train.reshape(NTRAIN, Lx,Lx,1)
x_test = x_test.reshape(NTEST,Lx,Lx,1)
print("x_train reshape: ", x_train.shape)

# Test loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=NBATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=NBATCH, shuffle=False)

# ----- test Loader-----
x_r, y_r = next(iter(train_loader))
Nb = x_r.size(0)
print('shapes x_r, y_r: ', x_r.shape, y_r.shape)

fig_file = os.path.join(FIG_DIR, 'sols_encoded.png')
stl = "Encoded Training"
make_plot_imgs(y_r, fig_file, show_fig=SHOW_FIG, stitle=stl, Nb=Nb, Nrows=4, Ncols=5, show_cbar=True)

# %%
# ----- Testing Model ------
import gan_fno2d
reload(gan_fno2d)
from gan_fno2d import Generator
G = Generator(MODES, MODES, WD, padding=PADDING).to(device)
print(G)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number Generator parameters: ", nn_params)
# ----- test the model-----
# one batch
x_r, y_r = next(iter(train_loader))
x_r = x_r.to(device)
print('x_r shape: ', x_r.shape)
y_g = G(x_r)
print('G(x_r) shape: ', y_g.shape)
# remove extra dim
y_g = y_g.squeeze()
y_g = y_g.detach().cpu()
fig_file = os.path.join(FIG_DIR, 'syn_imgs.png')
stl = "Model one pass"
make_plot_imgs(y_g, fig_file, show_fig=SHOW_FIG, stitle=stl, Nb=NBATCH, Nrows=4, Ncols=5, show_cbar=True)

#%%
# set up model and optimizers
G = Generator(MODES, MODES, WD, padding=PADDING).to(device)
# Create optimizers for the discriminator and generator
g_optimizer = Adam(G.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

fn_loss = LpLoss(size_average=False)

# %%
# train the network
G.train()
g_loss_train_v = np.zeros(NUM_EPOCHS)
g_loss_eval_v = np.zeros(NUM_EPOCHS)
y_normalizer.cuda()
x_normalizer.cuda()

for epoch in range(NUM_EPOCHS):
    # accumulator for losses
    g_train_loss = 0.0
    g_eval_loss = 0.0

    for batch_i, (x_r, y_r) in enumerate(train_loader):
        # --------- main training loop -------------
        # current batch size
        Nb = x_r.size(0)
        # move to GPU
        x_r = x_r.to(device)
        y_r = y_r.to(device)
        # clear gradiants
        g_optimizer.zero_grad()
        # forward pass
        y_g = G(x_r)
        # remove extra dim
        y_g = y_g.squeeze()
        # decode the output
        y_g = y_normalizer.decode(y_g)
        y_r = y_normalizer.decode(y_r)

        # calculate loss
        g_loss = fn_loss(y_g.view(Nb,-1), y_r.view(Nb,-1))
        # Calculate gradients
        g_loss.backward()
        # Update wights -> run optimizer
        # backprojection step
        g_optimizer.step()
        # store accumulated loss
        g_train_loss += g_loss.item()

        # Print some loss stats
        if batch_i % PRINT_EVERY == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | g_loss: {:6.4f}'.format(
                epoch + 1, NUM_EPOCHS, g_loss.item()))

        # ----- End training epoch ------
    # take scheduler step
    scheduler.step()
    G.eval()
    # Evaluate model
    with torch.no_grad():
        for (x_r, y_r) in test_loader:
            # batch size
            Nb = x_r.size(0)
            # move to GPU
            x_r = x_r.to(device)
            y_r = y_r.to(device)
            # forward pass
            y_g = G(x_r)
            # remove extra dim
            y_g = y_g.squeeze()
            # decode the output
            y_g = y_normalizer.decode(y_g)
            y_r = y_normalizer.decode(y_r)
            # calculate loss
            g_loss = fn_loss(y_g.view(Nb,-1), y_r.view(Nb,-1))
            # store accumulated loss
            g_eval_loss += g_loss.item()
            # End evaluation loop

        # --- plots use final validation iteration ---
        x_r = x_r.squeeze()
        x_r = x_normalizer.decode(x_r)
        x_r = x_r.detach().cpu().numpy()
        # model output
        y_g = y_g.detach().cpu().numpy()
        fig_file = os.path.join(FIG_DIR, f"solv_ep_{epoch}.png")
        stl = f"Solv to Poison 2D Epoch={epoch}"
        make_plots_sol2D(x_r, y_g, Np=4, fig_file=fig_file, stitle=stl, show_fig=False, show_cbar=True)

    # ----- save losses ----------
    g_loss_train_v[epoch] = g_train_loss / NTRAIN
    g_loss_eval_v[epoch] = g_eval_loss /NTEST
    # ----- back to train mode ----
    G.train()

# %%
# make lot of losses
iep = np.arange(NUM_EPOCHS) + 1.0
fig_file = os.path.join(OUT_DIR, f"solv_losses.png")
plt.figure(figsize=(8, 6))
plt.plot(iep, g_loss_train_v, color='C0', label="Training Loss")
plt.plot(iep, g_loss_eval_v, color='C1', label="Evaluation Loss")
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title("Losses FNO 2D ")
plt.savefig(fig_file, format='png')
# plt.show()
# %%
