#!/usr/bin/env python3

from dataclasses import dataclass
import itertools
import os
import shutil
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mlflow

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from thisquakedoesnotexist.utils.param_parser import ParamParser
from thisquakedoesnotexist.utils.plotting import plot_syn_data_grid
from thisquakedoesnotexist.utils.plotting import plot_syn_data_single
from thisquakedoesnotexist.utils.plotting import plot_waves_1C
from thisquakedoesnotexist.utils.random_fields import rand_noise, uniform_noise
from thisquakedoesnotexist.utils.tracking import log_model_mlflow
from thisquakedoesnotexist.utils.tracking import log_params_mlflow
from thisquakedoesnotexist.models.gan import Discriminator, Generator
from thisquakedoesnotexist.utils.data_utils import SeisData

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
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # create directory for figures
    syn_data_dir = os.path.join(output_dir, "syn_data")
    if not os.path.exists(syn_data_dir):
        os.makedirs(syn_data_dir)

    fig_dir = os.path.join(output_dir, "figs")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # create dir for stats
    tt_stats_dir = os.path.join(output_dir, "time_stats")
    if not os.path.exists(tt_stats_dir):
        os.makedirs(tt_stats_dir)

    # make a copy of gan.py which contains the model specifications
    model_file_name = model_file.split("/")[-1]
    gan_out = os.path.join(output_dir, model_file_name)
    shutil.copyfile(model_file, gan_out)
    return output_dir, models_dir, fig_dir, tt_stats_dir


@dataclass(init=True, repr=True)
class ConditionalVariables(object):
    """Dataclass to hold min, maxConditionalVariablesDocstring for ClassName."""
    dist_min: float
    dist_max: float
    dist_step_size: float
    dist_bins: np.ndarray
    mag_min: float
    mag_max: float
    mag_step_size: float
    mag_bins: np.ndarray
    vs30_min: float
    vs30_max: float
    vs30_step_size: float
    vs30_bins: np.ndarray


def get_cond_var_bins(dataset, num_bins=10, no_vs30=False):
    # Add slight padding to boundaries in both directions in order to include
    # values that land on the boundaries of the bins
    dist_min = dataset.vc_min["dist"] - 1e-5
    dist_max = dataset.vc_max["dist"] + 1e-5
    dist_step_size = (dist_max - dist_min) / num_bins
    dist_bins = np.arange(
        dist_min, dist_max + dist_step_size / 2.0, step=dist_step_size
    )

    mag_min = dataset.vc_min["mag"] - 1e-5
    mag_max = dataset.vc_max["mag"] + 1e-5
    mag_step_size = (mag_max - mag_min) / num_bins
    mag_bins = np.arange(
        mag_min, mag_max + mag_step_size / 2.0, step=mag_step_size
    )

    vs30_min = dataset.vc_min["vs30"] - 1e-5
    vs30_max = dataset.vc_max["vs30"] + 1e-5
    vs30_step_size = (vs30_max - vs30_min) / num_bins
    if no_vs30:
        vs30_step_size = vs30_max - vs30_min
    vs30_bins = np.arange(
        vs30_min, vs30_max + vs30_step_size / 2.0, step=vs30_step_size
    )
    return {'dist_bins': dist_bins, 'mag_bins': mag_bins, 'vs30_bins': vs30_bins}


def calc_mean_distances(dataset, wfs, c_norms, means, samples, noise_dim):
    c_norms = c_norms.reshape(-1, 1)
    real_data = wfs * c_norms
    real_data_mean = np.mean(real_data, axis=0)
    
    dist = means['dist']
    mag = means['mag']
    vs30 = means['vs30']

    dist_max = dataset.df_meta['dist'].max()
    mag_max = dataset.df_meta['mag'].max()
    vs30_max = dataset.df_meta['vs30'].max()

    vc_list = [
        dist / dist_max * torch.ones(samples, 1).cuda(),
        mag / mag_max * torch.ones(samples, 1).cuda(),
        vs30 / vs30_max * torch.ones(samples, 1).cuda(),
    ]

    grf = rand_noise(1, noise_dim, device=device)
    random_data = grf.sample(samples)
    syn_data, syn_scaler = G(random_data, *vc_list)
    syn_data = syn_data.squeeze().detach().cpu().numpy()

    syn_data = syn_data * syn_scaler.detach().cpu().numpy()

    sd_mean = np.mean(syn_data, axis=0)

    l2_dist = np.sum(np.abs(real_data_mean - sd_mean)) / len(sd_mean)
    mse = np.sum((real_data_mean - sd_mean)**2) / len(sd_mean)
    # print("L2: ", l2_dist)
    # print("MSE: ", mse)
    return (l2_dist, mse)


def plot_metrics_matrix(dataset, vc_bins, fig_dir):
    dist_bin_centers = []
    mag_bin_centers = []

    dist_bins = vc_bins['dist_bins']
    mag_bins = vc_bins['mag_bins']
    vs30_bins = vc_bins['vs30_bins']

    for i in range(10):
        dist_border = [dist_bins[i], dist_bins[i+1]]
        mag_border = [mag_bins[i], mag_bins[i+1]]
        wfs, c_norms, means, n_obs = get_waves_real_bin(dataset, dist_border, mag_border, vs30_bins)
        dist_bin_centers.append(means['dist'].round(1))
        mag_bin_centers.append(means['mag'].round(1))

    l2_mat = np.full((10, 10), np.nan)
    mse_mat = np.full((10, 10), np.nan)

    for i in range(10):
        for j in range(10):
            dist_border = [dist_bins[i], dist_bins[i+1]]
            mag_border = [mag_bins[j], mag_bins[j+1]]
            wfs, c_norms, means, n_obs = get_waves_real_bin(dataset, dist_border, mag_border, vs30_bins)
            
            if n_obs < 25:
                print(f"Bucket [{i}, {j}] only contains {n_obs} waveforms. Skipping ...")
                print(dist_border, mag_border)
                continue
            
            # plot_real_syn_bucket(wfs, c_norms, means, n_obs, dist_border, mag_border)
            l2, mse = calc_mean_distances(wfs, c_norms, means, n_obs, dist_border, mag_border)
            l2_mat[i, j] = l2
            mse_mat[i, j] = mse
            
    plt.title('Log-L2 Distance Real-Synthetic Data')
    ax = sns.heatmap(np.log(l2_mat), xticklabels=mag_bin_centers, yticklabels=dist_bin_centers)
    ax.invert_yaxis()
    plt.xlabel('Bin Mean Magnitude')
    plt.ylabel('Bin Mean Distance [km]')
    fig_file = os.path.join(fig_dir, f"l2_metric_matrix.png")
    plt.savefig(fig_file, format="png")

    plt.title('Log-MSE Distance Real-Synthetic Data')
    ax = sns.heatmap(np.log(mse_mat), xticklabels=mag_bin_centers, yticklabels=dist_bin_centers)
    ax.invert_yaxis()
    plt.xlabel('Bin Mean Magnitude')
    plt.ylabel('Bin Mean Distance [km]')
    fig_file = os.path.join(fig_dir, f"mse_metric_matrix.png")
    plt.savefig(fig_file, format="png")


def evaluate_model(G, n_waveforms, dataset, noise_dim, lt, dt, fig_dir):
    assert dataset.vc_min["dist"] == dataset.df_meta['dist'].min()

    cond_var_bins = get_cond_var_bins(dataset, 10, True)

    n_rows = 8
    n_cols = 4
    n_tot = n_rows * n_cols 

    dist_min = dataset.df_meta['dist'].min()
    dist_max = dataset.df_meta['dist'].max()
    dist_mean = dataset.df_meta['dist'].mean()

    mag_min = dataset.df_meta['mag'].min()
    mag_max = dataset.df_meta['mag'].max()
    mag_mean = dataset.df_meta['mag'].mean()
        
    vs30_mean = dataset.df_meta['vs30'].mean()
    vs30_max = dataset.df_meta['vs30'].max()
    
    dists = [dist_min, dist_mean, dist_max]
    mags = [mag_min, mag_mean, mag_max]

    for dist in dists:
        for mag in cond_var_bins['mag_bins']:
            vc_list = [
                dist / dist_max * torch.ones(samples, 1).cuda(),
                mag / mag_max * torch.ones(samples, 1).cuda(),
                vs30_mean / vs30_max * torch.ones(samples, 1).cuda(),
            ]

            grf = rand_noise(1, noise_dim, device=device)
            random_data = grf.sample(samples)
            syn_data, syn_scaler = G(random_data, *vc_list)
            syn_data = syn_data.squeeze().detach().cpu().numpy()
            syn_data = syn_data * syn_scaler.detach().cpu().numpy()
            
            synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
            sd_mean = np.mean(synthetic_data_log, axis=0)

            y = np.exp(sd_mean)

            nt = synthetic_data_log.shape[1]
            tt = dt * np.arange(0, nt)
            plt.semilogy(tt, y, '-' , label=f'Dist: {dist}km, Mag: {mag}', alpha=0.8, lw=0.5)

        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Log-Amplitude')
        fig_file = os.path.join(fig_dir, f"syn_data_mag_dependence_dist_{dist}.png")
        plt.savefig(fig_file, format="png")

    for mag in mags:
        for dist in cond_var_bins['dist_bins']:
            vc_list = [
                dist / dist_max * torch.ones(samples, 1).cuda(),
                mag / mag_max * torch.ones(samples, 1).cuda(),
                vs30_mean / vs30_max * torch.ones(samples, 1).cuda(),
            ]

            grf = rand_noise(1, noise_dim, device=device)
            random_data = grf.sample(samples)
            syn_data, syn_scaler = G(random_data, *vc_list)
            syn_data = syn_data.squeeze().detach().cpu().numpy()
            syn_data = syn_data * syn_scaler.detach().cpu().numpy()
            
            synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
            sd_mean = np.mean(synthetic_data_log, axis=0)

            y = np.exp(sd_mean)

            nt = synthetic_data_log.shape[1]
            tt = dt * np.arange(0, nt)
            plt.semilogy(tt, y, '-' , label=f'Dist: {dist}km, Mag: {mag}', alpha=0.8, lw=0.5)

        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Log-Amplitude')
        fig_file = os.path.join(fig_dir, f"syn_data_dist_dependence_mag_{dist}.png")
        plt.savefig(fig_file, format="png")

    for dist, mag in itertools.product(dists, mags):
        vc_list = [
                dist / dist_max * torch.ones(n_tot, 1).cuda(),
                mag / mag_max * torch.ones(n_tot, 1).cuda(),
                vs30_mean / vs30_max * torch.ones(n_tot, 1).cuda(),
        ]
        random_data = grf.sample(n_tot)
        syn_data, syn_scaler = G(random_data, *vc_list)
        syn_data = syn_data.squeeze()
        syn_data = np.array(syn_data.detach().cpu())
        syn_scaler = syn_scaler.detach().cpu().numpy()

        syn_data = syn_data * syn_scaler

        n_t = syn_data.shape[1]
        tt = dt * np.arange(0, n_t)

        plt.figure()
        fig, axs = plt.subplots(n_rows, n_cols, sharex='col',
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.15},
                                figsize=(36,12),
                                )

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude", labelpad=10.0)

        for i, ax in enumerate(axs.flat):
            # ax.set_ylim((-1, 1))
            # ax.set_aspect('equal', 'box')
            ax.plot(tt, syn_data[i, :], linewidth=0.5)
            low, high = ax.get_ylim()
            bound = max(abs(low), abs(high))
            ax.set_ylim(-bound, bound)
            
        # plt.tight_layout(pad=0.2)
        fig.suptitle(f'Randomly drawn samples from Generator. Dist: {dist:.1f} km, Mag: {mag:.1f}')
        fig_file = os.path.join(fig_dir, f"generated_data_{dist:.1f}_km_mag_{mag:.1f}.png")
        plt.savefig(fig_file, format="png")

        vs30 = vs30_mean
        samples = get_syntetic_data(
            G,
            n_waveforms,
            dataset,
            noise_dim,
            dist,
            mag,
            vs30
        )
        plot_syn_data_grid(samples, dt, dist, mag, vs30, fig_dir)
        plot_syn_data_single(samples, dt, lt, dist, mag, vs30, fig_dir)


def get_syntetic_data(G, n_waveforms, sdat_train, noise_dim, dist, mag, vs30):
    """get_syntetic_data returns n=n_waveforms number of synthetic waveforms for the corresponding conditional variables.

    _extended_summary_

    :param G: Generator object to create waveforms
    :type G: Generator
    :param n_waveforms: number of waveforms to create
    :type n_waveforms: int
    :param n_cond_bins: number of distance bins to use
    :type n_cond_bins: int
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

    dist_max = sdat_train.vc_max["dist"]
    mag_max = sdat_train.vc_max["mag"]
    vs30_max = sdat_train.vc_max["vs30"]

    vc_list = [
        dist / dist_max * torch.ones(samples, 1).cuda(),
        mag / mag_max * torch.ones(samples, 1).cuda(),
        vs30 / vs30_max * torch.ones(samples, 1).cuda(),
    ]

    grf = rand_noise(1, noise_dim, device=device)
    z = grf.sample(samples)

    G.eval()
    x_g, x_scaler = G(z, *vc_list)

    x_g = x_g.squeeze().detach().cpu()
    x_scaler = x_scaler.squeeze().detach().cpu()


    good_samples = []
    for wf, scaler in zip(x_g, x_scaler):
        tv = np.sum(np.abs(np.diff(wf)))
        # If generator sample fails to generate a seismic signal, skip it.
        # threshold value is emperically chosen
        if tv < 40:
            continue

        good_samples.append(wf * scaler)
    return good_samples[:n_waveforms]


def get_waves_real_bin(s_dat, distbs, mbs, vsb):
    # get dataframe with attributes
    df = s_dat.df_meta
    # get waves
    wfs = s_dat.wfs
    cnorms = s_dat.cnorms
    print(df.shape)
    # select bin of interest
    ix = (
        (distbs[0] <= df["dist"])
        & (df["dist"] < distbs[1])
        & (mbs[0] <= df["mag"])
        & (df["mag"] <= mbs[1])
        & (vsb[0] <= df["vs30"])
        & (df["vs30"] < vsb[1])
    )

    # get normalization coefficients
    df_s = df[ix]
    # get waveforms
    ws_r = wfs[ix, :, :]
    c_r = cnorms[ix, :]
    Nobs = ix.sum()
    print("# observations", Nobs)
    print("MAG: {:.2f}".format(df_s["mag"].min(), df_s["mag"].max()))
    print("DIST: {:.2f}".format(df_s["dist"].min(), df_s["dist"].max()))
    print("Vs30: {:.2f}".format(df_s["vs30"].min(), df_s["vs30"].max()))
    print("MAG Mean: {:.2f}".format(df_s["mag"].mean()))
    print("DIST Mean: {:.2f}".format(df_s["dist"].mean()))
    print("Vs30 Mean: {:.2f}".format(df_s["vs30"].mean()))

    return (ws_r, c_r)


def main():
    args = None
    args = ParamParser.parse_args(args=args)
    
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    log_params_mlflow(args)
    print("Tracking URI: ", mlflow.tracking.get_tracking_uri())
    print("Experiment name: ", args.experiment_name)

    run_id = mlflow.active_run().info.run_id[:8]
    print("MLFlow run ID: ", run_id)
    print(args)

    condv_names = ["dist", "mag", "vs30"]
    nbins_dict = {
        "dist": args.n_cond_bins,
        "mag": args.n_cond_bins,
        "vs30": 1 # TODO: implement argument for vs30 on/off ||args.n_cond_bins
    }

    out_dir, model_dir, fig_dir, tt_stats_dir = set_up_folders(args, run_id)
    print(f"Output directory: {out_dir}\nModel directory: {model_dir}")

    # total number of training samples
    f = np.load(args.data_file)
    n_samples = len(f)
    del f

    # get all indexes
    ix_all = np.arange(n_samples)
    # get training indexes
    n_train = int(n_samples * args.frac_train)
    ix_train = np.random.choice(ix_all, size=n_train, replace=False)
    ix_train.sort()
    # get validation indexes
    ix_val = np.setdiff1d(ix_all, ix_train, assume_unique=True)
    ix_val.sort()
    
    sdat_train = SeisData(
        data_file=args.data_file,
        attr_file=args.attr_file,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        v_names=condv_names,
        isel=ix_train,
    )

    sdat_val = SeisData(
        data_file=args.data_file,
        attr_file=args.attr_file,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        v_names=condv_names,
        isel=ix_val,
    )

    # Instatiate generator and discriminator
    D = Discriminator()
    G = Generator(z_size=args.noise_dim)
    print(D)
    print(G)

    if cuda:
        G.cuda()
        D.cuda()
    
    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(
        D.parameters(),
        lr=args.lr,
        betas=[args.beta1, args.beta2]
    )
    g_optimizer = optim.Adam(
        G.parameters(),
        lr=args.lr,
        betas=[args.beta1, args.beta2]
    )

    # keep track of loss and generated, "fake" samples
    losses_train = []
    losses_val = []

    # weigth for gradient penalty regunlarizer
    reg_lambda = args.gp_lambda

    # batch size
    batch_size = sdat_train.get_batch_size()

    n_train_btot = sdat_train.get_Nbatches_tot()
    n_val_btot = sdat_val.get_Nbatches_tot()


    print("Training Batches: ", n_train_btot)
    print("Validation Batches: ", n_val_btot)

    d_wloss_ep = np.zeros(args.epochs)
    d_total_loss_ep = np.zeros(args.epochs)
    g_loss_ep = np.zeros(args.epochs)

    # -> START TRAINING LOOP
    for i_epoch in range(args.epochs):
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

        n_critic = args.n_critic
        for i_batch in range(n_train_btot):
            for i_c in range(n_critic):
                ### ---------- DISCRIMINATOR STEP ---------------
                # 1. get real data
                # get random sample
                (data_b, ln_cb, i_vc) = sdat_train.get_rand_batch()
                # waves
                real_wfs = torch.from_numpy(data_b).float()
                # normalization constans
                real_lcn = torch.from_numpy(ln_cb).float()
                # conditional variables
                i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
                # number of samples
                Nsamp = real_wfs.size(0)
                # load into gpu
                if cuda:
                    real_wfs = real_wfs.cuda()
                    real_lcn = real_lcn.cuda()
                    i_vc = [i_v.cuda() for i_v in i_vc]

                # clear gradients
                d_optimizer.zero_grad()

                # 2. get fake waveforms
                # random gaussian noise
                z = uniform_noise(batch_size, args.noise_dim)
                z = torch.from_numpy(z).float()
                # move z to GPU, if available
                if cuda:
                    z = z.cuda()
                # generate a batch of waveform no autograd
                # important use same conditional variables
                (fake_wfs, fake_lcn) = G(z, *i_vc)

                # 3. compute regularization term for loss
                # random constant
                alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
                # make a view for multiplication
                alpha_cn = alpha.view(Nsamp, 1)
                # Get random interpolation between real and fake samples
                # for waves

                real_wfs = real_wfs.view(-1, 1, 1000, 1)
                real_lcn = real_lcn.view(real_lcn.size(0), -1)

                Xwf_p = (alpha * real_wfs + ((1.0 - alpha) * fake_wfs)).requires_grad_(
                    True
                )
                # for normalization
                Xcn_p = (
                    alpha_cn * real_lcn + ((1.0 - alpha_cn) * fake_lcn)
                ).requires_grad_(True)
                # apply dicriminator
                D_xp = D(Xwf_p, Xcn_p, *i_vc)
                # Get gradient w.r.t. interpolates waveforms
                Xout_wf = Variable(Tensor(Nsamp, 1).fill_(1.0), requires_grad=False)
                grads_wf = torch.autograd.grad(
                    outputs=D_xp,
                    inputs=Xwf_p,
                    grad_outputs=Xout_wf,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                grads_wf = grads_wf.view(grads_wf.size(0), -1)
                # get gradients w.r.t. normalizations
                Xout_cn = Variable(Tensor(Nsamp, 1).fill_(1.0), requires_grad=False)
                grads_cn = torch.autograd.grad(
                    outputs=D_xp,
                    inputs=Xcn_p,
                    grad_outputs=Xout_cn,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                # concatenate grad vectors
                grads = torch.cat(
                    [
                        grads_wf,
                        grads_cn,
                    ],
                    1,
                )

                # 4. Compute losses
                d_gp_loss = reg_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()
                d_w_loss = -torch.mean(D(real_wfs, real_lcn, *i_vc)) + torch.mean(
                    D(fake_wfs, fake_lcn, *i_vc)
                )
                d_loss = d_w_loss + d_gp_loss

                # 5. Calculate gradients
                d_loss.backward()
                # 6. update model weights -> run optimizer
                d_optimizer.step()

            ### ---------- END DISCRIMINATOR STEP ---------------
            # Get discriminator losses
            d_train_wloss = d_w_loss.item()
            d_train_gploss = d_gp_loss.item()

            ### -------------- TAKE GENERATOR STEP ------------------------
            # take a generator step every n_critic generator iterations
            # set initial gradients to zero
            g_optimizer.zero_grad()

            # 1. Train with fake waveforms

            # Generate fake waveforms
            z = uniform_noise(batch_size, args.noise_dim)
            z = torch.from_numpy(z).float()
            # get random sampling of conditional variables
            i_vg = sdat_train.get_rand_cond_v()
            i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
            # move to GPU
            if cuda:
                z = z.cuda()
                i_vg = [i_v.cuda() for i_v in i_vg]
            # forward step 1 -> generate fake waveforms
            (fake_wfs, fake_lcn) = G(z, *i_vg)
            # calculate loss
            g_loss = -torch.mean(D(fake_wfs, fake_lcn, *i_vg))

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
            losses_train.append((d_train_wloss, d_train_gploss, g_train_loss))

            # print after some iterations
            if i_batch % args.print_freq == 0:
                # append discriminator loss and generator loss
                # print discriminator and generator loss
                print(
                    "Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}".format(
                        i_epoch + 1, args.epochs, d_loss.item(), g_loss.item()
                    )
                )
            ### ------------- end batch --------------

        # ----- End training epoch ------
        # store losses
        losses_val.append((d_val_wloss, d_val_gploss, g_val_loss))

        # save generator
        fmodel = os.path.join(model_dir, "model_G_epoch_" + str(i_epoch) + ".pth")
        torch.save({"state_dict": G.state_dict()}, fmodel)

        # --------- End training epoch -----------
        # save losses
        # store train losses

        d_wloss_ep[i_epoch] = d_train_wloss / n_train
        d_total_loss_ep[i_epoch] = (d_train_wloss + d_train_gploss) / n_train
        g_loss_ep[i_epoch] = g_train_loss / n_train

        mlflow.log_metric(key="d_train_wloss", value=d_wloss_ep[i_epoch], step=i_epoch)
        mlflow.log_metric(
            key="d_total_loss", value=d_total_loss_ep[i_epoch], step=i_epoch
        )
        mlflow.log_metric(key="g_train_loss", value=g_loss_ep[i_epoch], step=i_epoch)

        distance = 60

        G.eval()
        z = uniform_noise(batch_size, args.noise_dim)
        z = torch.from_numpy(z).float()
        # get random sampling of conditional variables
        i_vg = sdat_train.get_rand_cond_v()
        i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
        # move to GPU
        if cuda:
            z = z.cuda()
            i_vg = [i_v.cuda() for i_v in i_vg]

        (x_g, fake_lcn) = G(z, *i_vg)

        x_g = x_g.squeeze()
        x_g = x_g.detach().cpu()
        fig_file = os.path.join(f"{out_dir}/syn_data", f"syn_ep_{i_epoch+1:05}.png")
        stl = f"Epoch: {i_epoch+1}, dist = {distance} km"
        plot_waves_1C(
            x_g,
            args.dt,
            ylim=(-1, 1),
            t_max=args.dt * args.lt,
            show_fig=False,
            fig_file=fig_file,
            stitle=stl,
        )
        # store model
        fmodel = os.path.join(model_dir, f"model_G_epoch_{i_epoch:05}.pth")
        torch.save({"state_dict": G.state_dict()}, fmodel)

        # plot validation statistics
        fig_file = os.path.join(tt_stats_dir, f"val_plots_{i_epoch}.png")

        # back to train mode
        G.train()

        mlflow.log_artifacts(f"{out_dir}/syn_data", "syn_data")

        # ----------- Validation Loop --------------
        G.eval()
        D.eval()
        for i_batch in range(n_val_btot):
            ### ---------- DISCRIMINATOR STEP ---------------

            # 1. get real data
            # get random sample
            (data_b, ln_cb, i_vc) = sdat_train.get_rand_batch()
            # waves
            real_wfs = torch.from_numpy(data_b).float()
            # normalization constans
            real_lcn = torch.from_numpy(ln_cb).float()
            # conditional variables
            i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
            # number of samples
            Nsamp = real_wfs.size(0)
            # load into gpu
            if cuda:
                real_wfs = real_wfs.cuda()
                real_lcn = real_lcn.cuda()
                i_vc = [i_v.cuda() for i_v in i_vc]

            # 2. get fake waveforms
            # random gaussian noise
            z = uniform_noise(batch_size, args.noise_dim)
            z = torch.from_numpy(z).float()
            # move z to GPU, if available
            if cuda:
                z = z.cuda()
            # generate a batch of waveform no autograd
            # important use same conditional variables
            (fake_wfs, fake_lcn) = G(z, *i_vc)

            # 3. compute regularization term for loss
            # random constant
            alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
            # make a view for multiplication
            alpha_cn = alpha.view(Nsamp, 1)

            real_wfs = real_wfs.view(-1, 1, 1000, 1)
            real_lcn = real_lcn.view(real_lcn.size(0), -1)

            # Get random interpolation between real and fake samples
            # for waves
            Xwf_p = (alpha * real_wfs + ((1.0 - alpha) * fake_wfs)).requires_grad_(True)
            # for normalization
            Xcn_p = (
                alpha_cn * real_lcn + ((1.0 - alpha_cn) * fake_lcn)
            ).requires_grad_(True)
            # apply dicriminator
            # Get gradient w.r.t. interpolates waveforms
            D_xp = D(Xwf_p, Xcn_p, *i_vc)
            Xout_wf = Variable(Tensor(Nsamp, 1).fill_(1.0), requires_grad=False)
            grads_wf = torch.autograd.grad(
                outputs=D_xp,
                inputs=Xwf_p,
                grad_outputs=Xout_wf,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads_wf = grads_wf.view(grads_wf.size(0), -1)
            # get gradients w.r.t. normalizations
            Xout_cn = Variable(Tensor(Nsamp, 1).fill_(1.0), requires_grad=False)
            grads_cn = torch.autograd.grad(
                outputs=D_xp,
                inputs=Xcn_p,
                grad_outputs=Xout_cn,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            # concatenate grad vectors
            grads = torch.cat(
                [
                    grads_wf,
                    grads_cn,
                ],
                1,
            )

            # 4. Compute losses
            d_gp_loss = reg_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()
            d_w_loss = -torch.mean(D(real_wfs, real_lcn, *i_vc)) + torch.mean(
                D(fake_wfs, fake_lcn, *i_vc)
            )
            d_loss = d_w_loss + d_gp_loss
            # use accumulators
            d_val_wloss += d_w_loss.item()
            d_val_gploss += d_gp_loss.item()
            ### ---------- END DISCRIMINATOR STEP ---------------

            ### ---------- TAKE GENERATOR STEP ------------------------

            # 1.  fake waveforms
            # Generate fake waveforms
            z = uniform_noise(batch_size, args.noise_dim)
            z = torch.from_numpy(z).float()
            # get random sampling of conditional variables
            i_vg = sdat_train.get_rand_cond_v()
            i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
            # move to GPU
            if cuda:
                z = z.cuda()
                i_vg = [i_v.cuda() for i_v in i_vg]
            # forward step 1 -> generate fake waveforms
            (fake_wfs, fake_lcn) = G(z, *i_vg)

            # calculate loss
            g_loss = -torch.mean(D(fake_wfs, fake_lcn, *i_vg))
            # use accumulator
            g_val_loss += g_loss.item()
            ### --------------  END GENERATOR STEP ------------------------
        # aggregate training losses
        d_val_wloss = d_val_wloss / n_val_btot
        d_val_gploss = d_val_gploss / n_val_btot
        g_val_loss = g_val_loss / n_val_btot

        mlflow.log_metric(key="d_val_wloss", value=d_val_wloss, step=i_epoch)
        mlflow.log_metric(key="d_val_gploss", value=d_val_gploss, step=i_epoch)
        mlflow.log_metric(key="g_val_loss", value=g_val_loss, step=i_epoch)

        # store losses
        losses_val.append((d_val_wloss, d_val_gploss, g_val_loss))
        ### --------- End Validation -------
        # AFTER EACH EPOCH #
        if i_epoch % args.print_freq == 0:
            # save generator
            fmodel = os.path.join(model_dir, f"model_G_epoch_{i_epoch:05}.pth")
            torch.save({"state_dict": G.state_dict()}, fmodel)
        if (i_epoch + 1) % 10 == 0:
            mlflow.pytorch.save_model(G, f"{out_dir}/model_epoch_{i_epoch + 1:05}")
            mlflow.pytorch.log_model(G, f"{out_dir}/model_epoch_{i_epoch + 1:05}")

    # back to train mode
    G.train()

    mlflow.log_artifacts(f"{out_dir}/syn_data", "syn_data")

    # make lot of losses
    iep = np.arange(1, 1 + args.epochs)
    fig_file = os.path.join(out_dir, f"figs/gan_losses.png")
    plt.figure(figsize=(8, 6))
    plt.plot(iep, d_wloss_ep, color="C0", label="W Distance")
    plt.plot(iep, d_total_loss_ep, color="C1", label="D Loss")
    plt.plot(iep, g_loss_ep, color="C2", label="G Loss")
    plt.legend()
    plt.ylabel("W Losses")
    plt.xlabel("Epoch")
    plt.title("Wasserstein GAN 1D Fourier, 1C")
    plt.savefig(fig_file, format="png")

    mlflow.pytorch.save_model(G, f"{out_dir}/model")
    mlflow.pytorch.log_model(G, f"{out_dir}/model")

    n_waveforms = 72 * 5

    evaluate_model(
        G,
        n_waveforms,
        args.n_cond_bins,
        sdat_train,
        args.noise_dim,
        args.lt,
        args.dt,
        fig_dir,
    )
    mlflow.log_artifacts(f"{out_dir}/figs", "figs")


if __name__ == "__main__":
    main()
