#!/usr/bin/env python3

import os
import shutil
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import mlflow

from thisquakedoesnotexist.utils.data import WaveDatasetDist
from thisquakedoesnotexist.condensed_code.dataUtils_v1 import SeisData

from thisquakedoesnotexist.utils.random_fields import rand_noise, uniform_noise
from thisquakedoesnotexist.utils.param_parser import ParamParser
from thisquakedoesnotexist.utils.plotting import plot_syn_data_grid, plot_syn_data_single
from thisquakedoesnotexist.utils.plotting import plot_waves_1C
from thisquakedoesnotexist.utils.tracking import log_model_mlflow, log_params_mlflow
from thisquakedoesnotexist.models.gan_d1 import Generator, Discriminator

SHOW_FIG = False
sns.set()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
print(f"Running on device: {device}")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def set_up_folders(conf_d, run_id):
    """
    Output configurations dictionary and create data directory
    """
    output_dir = conf_d.output_dir + "_" + str(run_id)
    model_file = conf_d.model_file

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create diretory for models
    models_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # create directory for figures
    syn_data_dir = os.path.join(output_dir, 'syn_data')
    if not os.path.exists(syn_data_dir):
        os.makedirs(syn_data_dir)

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


def evaluate_model(G, n_waveforms, n_dist_bins, dist, dataset, noise_dim, lt, dt, fig_dir):
    samples = get_syntetic_data(G, n_waveforms, n_dist_bins, dist, dataset, noise_dim, lt, dt)
    plot_syn_data_grid(samples, dist, dt, fig_dir)
    plot_syn_data_single(samples, dist, dt, lt, fig_dir)


def get_syntetic_data(G, n_waveforms, n_dist_bins, dist, dataset, noise_dim, lt, dt):
    """get_syntetic_data returns n=n_waveforms number of synthetic waveforms for the corresponding distance dist.

    _extended_summary_

    :param G: Generator object to create waveforms
    :type G: Generator
    :param n_waveforms: number of waveforms to create
    :type n_waveforms: int
    :param n_dist_bins: number of distance bins to use
    :type n_dist_bins: int
    :param dist: conditional variable for distance
    :type dist: float
    :param dataset: _description_
    :type dataset: WaveDatasetDist
    :param noise_dim: _description_
    :type noise_dim: int
    :param lt: _description_
    :type lt: int
    :param dt: _description_
    :type dt: float
    :return: list of len n_waveforms of synthetic waveforms
    :rtype: list
    """
    # Create some extra waveforms to filter out mode collapse
    samples = 2 * n_waveforms
    
    dist = 60.0
    mag = 5.8
    vs30 = 0.5

    vc_list = [dist * torch.ones(samples, 1).cuda(),
               mag * torch.ones(samples, 1).cuda(),
               vs30 * torch.ones(samples, 1).cuda(),
    ]
    
    # breakpoint()
    
    grf = rand_noise(1, noise_dim, device=device)
    # distv = dataset.fn_dist_scale(grf)
    #distv = torch.from_numpy(grf).float()
    # distv = distv.to(device)
    z = grf.sample(samples)

    G.eval()
    
    x_g = G(z, *vc_list)
    #x_g = G(z, distv)
    x_g = x_g.squeeze()
    x_g = x_g.detach().cpu()

    good_samples = []
    for wf in x_g:
        tv = np.sum(np.abs(np.diff(wf)))
        # If generator sample fails to generate a seismic signal, skip it
        # threshold value is emperically chosen 
        if tv < 40:
            continue
        good_samples.append(wf)
    return good_samples[:n_waveforms]


def get_waves_real_bin(s_dat, distbs, mbs, vsb):
    # get dataframe with attributes
    df = s_dat.df_meta
    # get waves
    wfs = s_dat.wfs
    cnorms = s_dat.cnorms
    print(df.shape)
    # select bin of interest
    ix = ((distbs[0] <= df['dist']) & (df['dist'] < distbs[1]) &
          (mbs[0] <= df['mag']) & (df['mag'] <= mbs[1]) &
          (vsb[0] <= df['vs30']) & (df['vs30'] < vsb[1]))

    # get normalization coefficients
    df_s = df[ix]
    # get waveforms
    ws_r = wfs[ix,:,:]
    c_r = cnorms[ix,:]
    Nobs = ix.sum()
    print('# observations', Nobs)
    print('MAG: {:.2f}'.format(df_s['mag'].min(), df_s['mag'].max()))
    print('DIST: {:.2f}'.format(df_s['dist'].min(), df_s['dist'].max()))
    print('Vs30: {:.2f}'.format(df_s['vs30'].min(), df_s['vs30'].max()))
    print('MAG Mean: {:.2f}'.format(df_s['mag'].mean()))
    print('DIST Mean: {:.2f}'.format(df_s['dist'].mean()))
    print('Vs30 Mean: {:.2f}'.format(df_s['vs30'].mean()))

    return (ws_r, c_r)


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
    print("MLFlow run ID: ", run_id)

    out_dir, model_dir, fig_dir, tt_stats_dir = set_up_folders(args, run_id)
    print(f'Output directory: {out_dir}\nModel directory: {model_dir}')

    dt = args.dt
    t_max = dt * args.lt

    f = np.load(args.data_file)
    num_samples = len(f)
    del f

    # get all indexes
    ix_all = np.arange(num_samples)

    condv_names = ['dist', 'mag', 'vs30']
    nbins_dict = {
        'dist': 30,
        'mag': 30,
        'vs30': 1,
    }

    # prepare data loader
    sdat_train = SeisData(
        data_file=args.data_file,
        attr_file=args.attr_file,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        v_names=condv_names,
        nbins_d=nbins_dict,
        isel=ix_all,
    )
    # dataset = WaveDatasetDist(data_file=args.data_file, attr_file=args.attr_file, ndist_bins=args.ndist_bins, dt=dt)
    # create train loader
    # train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # batch size
    Nbatch = sdat_train.get_batch_size()
    # total number of batches
    N_train_btot = sdat_train.get_Nbatches_tot()

    # n_train = len(dataset)
    
    # x_r, y_r = next(iter(train_loader))
    # n_samples = x_r.size(0)
    # n_plots = n_samples

    n_channels = 1

    grf = rand_noise(n_channels, args.noise_dim, device=device)
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

    D.train()
    G.train()
    d_wloss_ep = np.zeros(args.epochs)
    d_total_loss_ep = np.zeros(args.epochs)
    g_loss_ep = np.zeros(args.epochs)

    # distance = 60.0
    # distv = distance * np.ones((n_samples, 2))
    # distv = dataset.fn_dist_scale(distv)
    # distv = torch.from_numpy(distv).float()
    # distv = distv.to(device)

    for ix_ep in range(args.epochs):
        d_train_wloss = 0.0
        d_train_gploss = 0.0
        g_train_loss = 0.0
        n_critic = 0
        
        for i_c in range(args.n_critic):
    
            (data_b, i_vc) = sdat_train.get_rand_batch()
            real_wfs = torch.from_numpy(data_b).float()
            real_wfs = real_wfs.view(-1, 1, 1000, 1)
            i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
            # number of samples
            n_samples = real_wfs.size(0)
            # load into gpu
            if cuda:
                real_wfs = real_wfs.cuda()
                i_vc = [i_v.cuda() for i_v in i_vc]


            # clear gradients
            d_optimizer.zero_grad()
            # 2. Get fake data
            # Gaussian random field
            z = grf.sample(n_samples)
            
            # breakpoint()

            fake_wfs = G(z, *i_vc)
            # 3. Compute gradient penalty term
            
            alpha = torch.rand(n_samples, 1, 1, 1, device=device)
            # Get random interpolation between real and fake samples

            # breakpoint()

            Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
            # apply discriminator
            D_xp = D(Xp, *i_vc)
            # Jacobian related variable
            Xout = Variable(torch.ones(n_samples,1, device=device), requires_grad=False)
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
            d_gp_loss = args.gp_lambda * ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
            d_w_loss = -torch.mean(D(real_wfs, *i_vc)) + torch.mean(D(fake_wfs, *i_vc))
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
                z = uniform_noise(Nbatch, args.noise_dim)
                z = torch.from_numpy(z).float()
                # get random sampling of conditional variables
                i_vg = sdat_train.get_rand_cond_v()
                i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
                # move to GPU
                if cuda:
                    z = z.cuda()
                    i_vg = [i_v.cuda() for i_v in i_vg]
                
                z = grf.sample(n_samples)
                fake_wfs = G(z, *i_vg)

                x_g = G(z, *i_vc)
                # 2. calculate loss
                g_loss = -torch.mean( D(fake_wfs, *i_vg) )
                
                # breakpoint()
                
                # The weights of the generator are optimized with respect
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
            if (i_c % args.print_freq == 0):
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_tot_loss: {:6.4f} | g_wloss: {:6.4f}'.format(
                    ix_ep + 1, args.epochs, d_loss.item(), d_w_loss.item()))

        # --------- End training epoch -----------
        # save losses
        d_wloss_ep[ix_ep] = d_train_wloss / N_train_btot
        d_total_loss_ep[ix_ep] = (d_train_wloss + d_train_gploss) / N_train_btot
        g_loss_ep[ix_ep] = g_train_loss / N_train_btot

        mlflow.log_metric(key="d_train_wloss", value=d_wloss_ep[ix_ep], step=ix_ep)
        mlflow.log_metric(key="d_total_loss", value=d_total_loss_ep[ix_ep], step=ix_ep)
        mlflow.log_metric(key="g_train_loss", value=g_loss_ep[ix_ep], step=ix_ep)

        # --- AFTER EACH EPOCH ---
        # generate and save sample synthetic images
        # eval mode for generating samples
        distance = 60
        
        G.eval()
        z = grf.sample(n_samples)
        x_g = G(z, *i_vc)
        x_g = x_g.squeeze()
        x_g = x_g.detach().cpu()
        fig_file = os.path.join(f'{out_dir}/syn_data', f"syn_ep_{ix_ep+1:03}.png")
        stl = f"Epoch: {ix_ep+1}, dist = {distance} km"
        plot_waves_1C(x_g, dt, ylim=(-1, 1), t_max=t_max, show_fig=False, fig_file=fig_file, stitle=stl)
        # store model
        fmodel = os.path.join(model_dir, f'model_G_epoch_{ix_ep:03}.pth')
        torch.save({'state_dict': G.state_dict()}, fmodel)

        # plot validation statistics
        fig_file = os.path.join(tt_stats_dir, f"val_plots_{ix_ep}.png")
        
        # back to train mode
        G.train()

        mlflow.log_artifacts(f"{out_dir}/syn_data", "syn_data")

    # make lot of losses
    iep = np.arange(1, 1 + args.epochs)
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

    n_waveforms = 72 * 5
    dist = 60

    evaluate_model(G, n_waveforms, args.ndist_bins, dist, sdat_train, args.noise_dim, args.lt, args.dt, fig_dir)
    mlflow.log_artifacts(f"{out_dir}/figs", "figs")
    
if __name__ == '__main__':
    main()