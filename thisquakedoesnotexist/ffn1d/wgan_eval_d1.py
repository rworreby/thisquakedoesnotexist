# %%
import os
import json
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# isaiimportss
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from imp import reload


# pytorch importsa
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# costum utilities
import dataUtils_v1
import gan_d1

# make sure class ais reloaded everytime the cell is executed
reload(dataUtils_v1)
from dataUtils_v1 import SeisData, plot_waves_3C

reload(gan_d1)
from gan_g1 import Discriminator, Generator

# -------- Input Parameters -------
config_d = {
    'data_file': '/scratch/mflorez/gmpw/train1x_set/train_P_M5/downsamp_1x_sel.npy',
    'attr_file': '/scratch/mflorez/gmpw/train1x_set/train_P_M5/wforms_table_sel.csv',
    'batch_size': 128,
    'noise_dim': 100,
    'epochs': 80,
    'sample_rate': 100.0,
    # the frasction of data for trainings-
    'frac_train': 0.8,

    # conditionals variablesdd
    # must match the header names in the table wforms_table.csv
    'condv_names': ['dist', 'mag', 'vs30'],
    # definitions of bins to use
    # either and integer with the number of bins
    # or aa list with the bin edgess
    'nbins_dict': {
        'dist': 30,
        'mag': 30,
        'vs30': 20,
    },

    # ---- wassertddain gan configuration -----
    # number of iterations for critics
    'critic_iter': 12,
    'gp_lambda': 10.0,
    'lr':  1e-4,
    # Adam optimizser parasmeters :
    'beta1': 0.0,
    'beta2': 0.9,

    # lowear and aaupper value to clip discriminator weights
    #'clip_value': 0.01, don't apply if using regularizer dd

    # ------------------- output onfiiguration ------------------
    'out_dir': '/scratch/mflorez/gmpw/torch-awGAN/P_d2_c12v40',
    'print_every': 400,
    'save_every': 1,
}

# acreate a fol
def init_gan_conf(conf_d):
    """
    Output configurations dictionary and ddcreate data directory
    """
    out_dir = conf_d['out_dir']

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

    # make a copy of gan.py which contains the model specifications
    gan_file = os.path.join('./', 'gan_g5.py')
    gan_out = os.path.join(out_dir, 'gan_models.py')
    shutil.copyfile(gan_file, gan_out)
    return out_dir, models_dir

# Get paths
OUT_DIR, MODEL_DIR = init_gan_conf(config_d)
print(OUT_DIR, MODEL_DIR)
print(config_d)

#%%
Ntot = 55151
frac = config_d['frac_train']
Nbatch = config_d['batch_size']

# get all indexes
ix_all = np.arange(Ntot)
# get training indexes
Nsel = int(Ntot*frac)
ix_train = np.random.choice(ix_all, size=Nsel, replace=False)
ix_train.sort()
# get validation indexes
ix_val = np.setdiff1d(ix_all, ix_train, assume_unique=True)
ix_val.sort()

# get and instance of data loader
sdat_train = SeisData(config_d['data_file'], attr_file=config_d['attr_file'], batch_size=Nbatch, sample_rate=config_d['sample_rate'], v_names=config_d['condv_names'], nbins_d=config_d['nbins_dict'], isel=ix_train)

print('total Train:', sdat_train.get_Ntrain())

sdat_val = SeisData(config_d['data_file'], attr_file=config_d['attr_file'], batch_size=Nbatch, sample_rate=config_d['sample_rate'], v_names=config_d['condv_names'], nbins_d=config_d['nbins_dict'], isel=ix_val)

print('total Validation:', sdat_val.get_Ntrain())
# get a random sample
(wfs, i_vc) = sdat_val.get_rand_batch()
print('shape:', wfs.shape)

#%%
# # ------- testing ---------
# dt = sdat_train.dt
# # get a random sample
# (wfs, i_vc) = sdat_train.get_rand_batch()
# print('shape:', wfs.shape)
# plot_waves_3C(wfs, dt, t_max=40.0, color='C0')
# print("dt: ", dt)
# %%

# generic tensor:
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def noise(Nbatch, dim):
    # Generate noise from a uniform distribution
    m = 3
    return np.random.normal(size=[Nbatch, m, dim]).astype(
        dtype=np.float32)

# ------- TRAIN MODELS --------

# ->
# -> training hyperparams
num_epochs = config_d['epochs']
# how often to print samples
print_every = config_d['print_every']
# number of noise variables
z_size = config_d['noise_dim']

# wassertain params
# iterations for critic
n_critic = config_d['critic_iter']

# optimizer parameters default values
lr = config_d['lr']
beta1 = config_d['beta1']
beta2 = config_d['beta2']

#%%
import gan_d1
reload(gan_d1)
from gan_d1 import Discriminator, Generator

# ->
# -> instatiate generator and discriminator

D = Discriminator()
G = Generator(z_size=z_size)
print(D)
print(G)

# ->
# -> train on GPU if available

if cuda:
    # move models to GPU
    G.cuda()
    D.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=[beta1,beta2])
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=[beta1,beta2])


# # --- uncomment for simple tests -----
#
# get random sample
(data_b, i_vc) = sdat_train.get_rand_batch()
real_wfs = torch.from_numpy(data_b).float()
i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
if cuda:
    real_wfs = real_wfs.cuda()
    i_vc = [i_v.cuda() for i_v in i_vc]
D_out = D(real_wfs, *i_vc)
print('shape D_out: ', D_out.shape)
#%%
# #%%
# #
# test generator
print('-- Training ----')
(wfs, i_vc) = sdat_train.get_rand_batch()
z = noise(Nbatch, z_size)
z = torch.from_numpy(z).float()
i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
if cuda:
    z = z.cuda()
    i_vc = [i_v.cuda() for i_v in i_vc]
fake_wfs = G(z,*i_vc)
print('shape G_out', fake_wfs.shape)

#%%
# ----- test gradient penalty -----

# gradient penalty
Nsamp = real_wfs.size(0)
alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
# Get random interpolation between real and fake samples
Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
# apply dicriminator
D_xp = D(Xp, *i_vc)
Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
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

# %%
# ->
# -> Main Training Loop
# keep track of loss and generated, "fake" samples
losses_train = []
losses_val = []

# wiegth for gradient penalty regunlarizer
reg_lambda = config_d['gp_lambda']

# batch size
Nbatch = sdat_train.get_batch_size()
# total number of batches
N_train_btot = sdat_train.get_Nbatches_tot()
N_val_btot = sdat_val.get_Nbatches_tot()

# print batch numbers
print('Training Batches: ', N_train_btot)
print('Validation Batches: ', N_val_btot)

# ->
# -> START TRAINING LOOP
for i_epoch in range(num_epochs):
    # store train losses
    d_train_wloss = 0.0
    d_train_gploss = 0.0
    g_train_loss = 0.0
    # store val losses
    d_val_wloss = 0.0
    d_val_gploss = 0.0
    g_val_loss = 0.0

    # ----- Training loop ------
    G.train()
    D.train()
    for i_batch in range(N_train_btot):
        for i_c in range(n_critic):
            ### ---------- DISCRIMINATOR STEP ---------------
            # 1. get real data
            # get random sample
            (data_b, i_vc) = sdat_train.get_rand_batch()
            real_wfs = torch.from_numpy(data_b).float()
            i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
            # number of samples
            Nsamp = real_wfs.size(0)
            # load into gpu
            if cuda:
                real_wfs = real_wfs.cuda()
                i_vc = [i_v.cuda() for i_v in i_vc]

            # clear gradients
            d_optimizer.zero_grad()

            # 2. get fake waveforms

            # random gaussian noise
            z = noise(Nbatch, z_size)
            z = torch.from_numpy(z).float()
            # move z to GPU, if available
            if cuda:
                z = z.cuda()
            # generate a batch of waveform no autograd
            fake_wfs = G(z,*i_vc)

            # 3. compute regularization term for loss
            alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
            # Get random interpolation between real and fake samples
            Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
            # apply dicriminator
            D_xp = D(Xp, *i_vc)
            Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
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
            d_gp_loss = reg_lambda*((grads.norm(2, dim=1) - 1) ** 2).mean()
            d_w_loss = -torch.mean(D(real_wfs, *i_vc)) + torch.mean(D(fake_wfs, *i_vc))
            d_loss = d_w_loss + d_gp_loss

            # 5. Calculate gradients
            d_loss.backward()
            # 6. update model weights -> run optimizer
            d_optimizer.step()

        ### ---------- END DISCRIMINATOR STEP ---------------
        # Get losses discriminator losses
        d_train_wloss = d_w_loss.item()
        d_train_gploss = d_gp_loss.item()

        ### -------------- TAKE GENERATOR STEP ------------------------
        # take a generator step every n_critic generator iterations
        # set initial gradients to zero
        g_optimizer.zero_grad()

        # 1. Train with fake waveforms

        # Generate fake waveforms
        z = noise(Nbatch, z_size)
        z = torch.from_numpy(z).float()
        # get random sampling of conditional variables
        i_vg = sdat_train.get_rand_cond_v()
        i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
        # move to GPU
        if cuda:
            z = z.cuda()
            i_vg = [i_v.cuda() for i_v in i_vg]
        # forward step 1 -> generate fake waveforms
        fake_wfs = G(z, *i_vg)
        # calculate loss
        g_loss = -torch.mean( D(fake_wfs, *i_vg) )

        # The wights of the generator are optimized with respect
        # to the discriminator loss
        # the generator is trained to produce data that will be classified
        # by the discriminator as "real"

        # Calculate gradients for generator
        g_loss.backward()
        # update weights for generator -> run optimizer
        g_optimizer.step()
        # store losses
        g_train_loss = g_loss.item()
        ### --------------  END GENERATOR STEP ------------------------
        # store losses
        losses_train.append((d_train_wloss, d_train_gploss, g_train_loss) )

        # print after some iterations
        if i_batch % print_every == 0:
            # append discriminator loss and generator loss
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                i_epoch + 1, num_epochs, d_loss.item(), g_loss.item()))
        ### ------------- end batch --------------
    # ----- End training epoch ------


    # ----------- Validation Loop --------------
    G.eval()
    D.eval()
    for i_batch in range(N_val_btot):

        ### ---------- DISCRIMINATOR STEP ---------------

        # 1. get real data
        # get random sample
        (data_b, i_vc) = sdat_val.get_rand_batch()
        real_wfs = torch.from_numpy(data_b).float()
        i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
        # number of samples
        Nsamp = real_wfs.size(0)
        # load into gpu
        if cuda:
            real_wfs = real_wfs.cuda()
            i_vc = [i_v.cuda() for i_v in i_vc]

        # 2. get fake waveforms

        # random gaussian noise
        z = noise(Nbatch, z_size)
        z = torch.from_numpy(z).float()
        # move z to GPU, if available
        if cuda:
            z = z.cuda()
        # generate a batch of waveform no autograd
        fake_wfs = G(z,*i_vc)

        # 3. compute regularization term for loss
        alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
        # Get random interpolation between real and fake samples
        Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
        # apply dicriminator
        D_xp = D(Xp, *i_vc)
        Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
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
        d_gp_loss = reg_lambda*((grads.norm(2, dim=1) - 1) ** 2).mean()
        d_w_loss = -torch.mean(D(real_wfs, *i_vc)) + torch.mean(D(fake_wfs, *i_vc))
        d_loss = d_w_loss + d_gp_loss

        d_val_wloss += d_w_loss.item()
        d_val_gploss += d_gp_loss.item()
        ### ---------- END DISCRIMINATOR STEP ---------------


        ### ---------- TAKE GENERATOR STEP ------------------------

        # 1. fake waveforms

        # Generate fake waveforms
        z = noise(Nbatch, z_size)
        z = torch.from_numpy(z).float()
        # get random sampling of conditional variables
        i_vg = sdat_val.get_rand_cond_v()
        i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
        # move to GPU
        if cuda:
            z = z.cuda()
            i_vg = [i_v.cuda() for i_v in i_vg]
        # forward step 1 -> generate fake waveforms
        fake_wfs = G(z, *i_vg)
        # calculate loss
        g_loss = -torch.mean( D(fake_wfs, *i_vg) )
        g_val_loss += g_loss.item()
            ### --------------  END GENERATOR STEP ------------------------
    # aggregate training losses
    d_val_wloss = d_val_wloss/N_val_btot
    d_val_gploss = d_val_gploss/N_val_btot
    g_val_loss = g_val_loss/N_val_btot
    # store losses
    losses_val.append((d_val_wloss, d_val_gploss, g_val_loss) )
    ### --------- End Validation -------
    # AFTER EACH EPOCH #
    if i_epoch % config_d['save_every'] == 0:
        # save generator
        fmodel = os.path.join(MODEL_DIR, 'model_G_epoch_'+str(i_epoch)+'.pth')
        torch.save({'state_dict': G.state_dict()}, fmodel)



# Save losses
ftrainl = os.path.join(OUT_DIR, 'train_losses.pkl')

with open(ftrainl, 'wb') as f:
    pkl.dump(losses_train, f)

fvall = os.path.join(OUT_DIR, 'val_losses.pkl')
with open(fvall, 'wb') as f:
    pkl.dump(losses_val, f)


print('Done!!')
# %%

