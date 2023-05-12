#!/usr/bin/env python3

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
from argparse import ArgumentParser
from importlib import reload
# ------ optimizers ------
import torch.optim as optim
from torchinfo import summary

# ------ optimizers ------

from torch.utils.data import DataLoader
from torch.autograd import Variable

# -------- tracking -------
import mlflow

# ---- custom utilities ---
from thisquakedoesnotexist.utils.data_utils import WaveDatasetDist
from thisquakedoesnotexist.utils.random_fields import *
from thisquakedoesnotexist.utils.utils import *
from thisquakedoesnotexist.utils.param_parser import ParamParser
from thisquakedoesnotexist.utils import tracking
from thisquakedoesnotexist.condensed_code.gan1d import Generator, Discriminator
# from thisquakedoesnotexist.plotting.fn_plot import *


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
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Running on device: {device}")


# setup folders to store experiments
def init_gan_conf(conf_d, run_id):
    """
    Output configurations dictionary and create data directory
    """
    output_dir = conf_d.output_dir + "_" + str(run_id)
    model_file = conf_d.model_file

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fname = os.path.join(output_dir, 'config_d.json')
    
    # create diretory for models
    models_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    # create directory for figures
    fig_dir = os.path.join(output_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # create dir for stats
    tt_stats_dir = os.path.join(output_dir, 'time_stats')
    if not os.path.exists(tt_stats_dir):
        os.makedirs(tt_stats_dir)

    # make a copy of gan.py which contains the model specifications
    model_file_name = model_file.split('/')[-1]
    gan_out = os.path.join(output_dir, model_file_name)
    shutil.copyfile(model_file, gan_out)
    return output_dir, models_dir, fig_dir, tt_stats_dir


def log_params_mlflow(params):
    """log_params_mlflow add parameters to mlflow tracking. 

    :param params: argument parser instance
    :type params: ParamParser
    """
    mlflow.log_param("Model_file", params.model_file)
    mlflow.log_param("Data_file", params.data_file)
    mlflow.log_param("Attribute_file", params.attr_file)
    mlflow.log_param("Learning_rate", params.lr)
    mlflow.log_param("Discriminator_input_size", params.lt)
    mlflow.log_param("Generator_noise_dimension", params.noise_dim)
    mlflow.log_param("GP_Lambda", params.gp_lambda)
    mlflow.log_param("Critic_iterations_per_training_cycle ", params.n_critic)
    mlflow.log_param("Beta_1", params.beta1)
    mlflow.log_param("Beta_2", params.beta2)
    mlflow.log_param("Epochs", params.epochs)
    mlflow.log_param("Batch_size", params.batch_size)


def log_model_mlflow(D, G, out_dir):
    with open(f'{out_dir}/generator.txt', 'w') as f:
        f.write(str(summary(G)))
    mlflow.log_artifact(f'{out_dir}/generator.txt', "Generator")
    
    with open(f'{out_dir}/discriminator.txt', 'w') as f:
        f.write(str(summary(D)))
    mlflow.log_artifact(f'{out_dir}/discriminator.txt', "Discriminator")
    
    with open(f'{out_dir}/generator_state_dict.txt', 'w') as f:
        f.write(str(G.state_dict()))
    mlflow.log_artifact(f'{out_dir}/generator_state_dict.txt', "Generator state dict")

    with open(f'{out_dir}/discriminator_state_dict.txt', 'w') as f:
        f.write(str(D.state_dict()))
    mlflow.log_artifact(f'{out_dir}/discriminator_state_dict.txt', "Discriminator state dict")    
    
    mlflow.log_param("Generator_num_params", sum(p.numel() for p in G.parameters() if p.requires_grad))
    mlflow.log_param("Discriminator_num_params", sum(p.numel() for p in D.parameters() if p.requires_grad))


def main():
    tracking_uri = '/home/rworreby/thisquakedoesnotexist/mlruns/'
    mlflow.set_tracking_uri(tracking_uri)

    print("Tracking URI: ", mlflow.tracking.get_tracking_uri())

    experiment_name = "Florez_CGAN"
    print("Experiment name: ", experiment_name)
    mlflow.set_experiment(experiment_name)
    

    args = None
    args = ParamParser.parse_args(args=args)
    print(args)

    log_params_mlflow(args)
    run_id = mlflow.active_run().info.run_id[:8]
    print("MLFLOW RUNID: ", run_id)

    # Get paths
    out_dir, model_dir, fig_dir, tt_stats_dir = init_gan_conf(args, run_id)
    print(f'Output directory: {out_dir}\nModel directory: {model_dir}')

    # Dataset sampling intervals
    dt = 0.04
    t_max = dt * args.lt

    # prepare data loader
    dataset = WaveDatasetDist(data_file=args.data_file, attr_file=args.attr_file, ndist_bins=args.ndist_bins, dt=dt)
    # create train loader
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n_train = len(dataset)
    
    x_r, y_r = next(iter(train_loader))
    Nb = x_r.size(0)
    num_plots = Nb

    grf = rand_noise(1, args.noise_dim, device=device)
    z = grf.sample(args.nplt_max)
    print("z shape: ", z.shape)
    z = z.squeeze()
    z = z.detach().cpu()

    # create new model instances
    D = Discriminator().to(device)
    G = Generator(z_size=args.noise_dim).to(device)

    d_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])

    log_model_mlflow(D, G, out_dir)
    
    mlflow.log_param("Generator_optimizer", g_optimizer)
    mlflow.log_param("Discriminator_optimizer", d_optimizer)

    # train the network
    D.train()
    G.train()
    d_wloss_ep = np.zeros(args.epochs)
    d_total_loss_ep = np.zeros(args.epochs)
    g_loss_ep = np.zeros(args.epochs)

    # distance array
    distance = 60.0
    distv = distance * np.ones((Nb, 1))
    distv = dataset.fn_dist_scale(distv)
    distv = torch.from_numpy(distv).float()
    distv = distv.to(device)

    for ix_ep in range(args.epochs):
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
            d_gp_loss = args.gp_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()
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
            if n_critic == args.n_critic:
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
                g_train_loss += args.n_critic * g_loss.item()

                # resete critic iterations to zero
                n_critic = 0

                ### --------------  END GENERATOR STEP ------------------------

            # Print some loss stats
            if (batch_i % args.print_freq==0):
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_tot_loss: {:6.4f} | g_wloss: {:6.4f}'.format(
                    ix_ep + 1, args.epochs, d_loss.item(), d_w_loss.item()))

        # --------- End training epoch -----------
        # save losses
        d_wloss_ep[ix_ep] = d_train_wloss / n_train
        d_total_loss_ep[ix_ep] = (d_train_wloss + d_train_gploss) / n_train
        g_loss_ep[ix_ep] = g_train_loss / n_train

        mlflow.log_metric(key="d_train_wloss", value=d_wloss_ep[ix_ep], step=ix_ep)
        mlflow.log_metric(key="d_total_loss", value=d_total_loss_ep[ix_ep], step=ix_ep)
        mlflow.log_metric(key="g_train_loss", value=g_loss_ep[ix_ep], step=ix_ep)

        # --- AFTER EACH EPOCH ---
        # generate and save sample synthetic images
        # eval mode for generating samples
        G.eval()
        z = grf.sample(num_plots)
        x_g = G(z,distv)
        x_g = x_g.squeeze()
        x_g = x_g.detach().cpu()
        fig_file = os.path.join(fig_dir, f"syn_ep_{ix_ep:03}.png")
        stl = f"Epoch: {ix_ep}, dist = {distance} km"
        plot_waves_1C(x_g, dt, t_max=t_max, show_fig=False, fig_file=fig_file, stitle=stl, color='C1')
        # store model
        fmodel = os.path.join(model_dir, f'model_G_epoch_{ix_ep:03}.pth')
        torch.save({'state_dict': G.state_dict()}, fmodel)

        # plot validation statistics
        fig_file = os.path.join(tt_stats_dir, f"val_plots_{ix_ep}.png")
        
        # back to train mode
        G.train()

        mlflow.log_artifacts(f"{out_dir}/figs", "figs")

    # make lot of losses
    iep = np.arange(args.epochs) + 1.0
    fig_file = os.path.join(out_dir, f"figs/gan_losses.png")
    plt.figure(figsize= (8,6))
    plt.plot(iep, d_wloss_ep, color='C0',label="W Distance")
    plt.plot(iep, d_total_loss_ep, color='C1',label="D Loss")
    plt.plot(iep, g_loss_ep, color='C2', label="G Loss")
    plt.legend()
    plt.ylabel('W Losses')
    plt.xlabel('Epoch')
    plt.title("Wasserstein GAN 1D Fourier, 1C")
    plt.savefig(fig_file, format='png')

    mlflow.pytorch.save_model(G, f"{out_dir}/model")
    mlflow.pytorch.log_model(G, f"{out_dir}/model")

if __name__ == '__main__':
    main()