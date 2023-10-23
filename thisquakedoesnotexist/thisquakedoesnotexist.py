#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mlflow
import torch
from torch.autograd import Variable
import torch.optim as optim

from utils.evaluation import evaluate_model
from models.gan import Discriminator, Generator
from utils.param_parser import ParamParser
from utils.plotting import plot_waves_1C
# from utils.plotting import plot_real_syn_bucket
from utils.random_fields import rand_noise, uniform_noise
from utils.tracking import log_params_mlflow
from utils.data_utils import SeisData, set_up_folders


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
print(f"Running on device: {device}")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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

    condv_names = ["dist", "mag"]

    dirs = set_up_folders(run_id, args)
    print(f"Output directory: {dirs['output_dir']}\nModel directory: {dirs['models_dir']}")

    # total number of training samples
    f = np.load(args.data_file)
    n_samples = len(f)
    del f

    # get all indexes
    ix_all = np.arange(n_samples)
    # get training indexes
    n_train = int(n_samples * args.frac_train)
    ix_train = np.random.choice(ix_all, size=n_train, replace=False, )
    ix_train.sort()
    # get validation indexes
    ix_val = np.setdiff1d(ix_all, ix_train, assume_unique=True)
    ix_val.sort()
    
    mlflow.log_param("Training Indices", ix_train)
    mlflow.log_param("Validation Indices", ix_val)

    sdat_all = SeisData(
        data_file=args.data_file,
        attr_file=args.attr_file,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        v_names=condv_names,
        isel=ix_all,
    )

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
        lr=args.learning_rate,
        betas=[args.beta1, args.beta2],
        # weight_decay=0.01,
    )
    g_optimizer = optim.Adam(
        G.parameters(),
        lr=args.learning_rate,
        betas=[args.beta1, args.beta2],
        # weight_decay=0.01,
    )

    losses_train = []
    losses_val = []

    # weigth for gradient penalty regularizer
    reg_lambda = args.gp_lambda

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

        # TODO: REMOVE THIS AGAIN
        n_train_btot = 1
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

        x_g = x_g.squeeze().detach().cpu().numpy()
        x_g = x_g * fake_lcn.detach().cpu().numpy()
        fig_file = os.path.join(f"{dirs['training_dir']}", f"syn_ep_{i_epoch+1:05}.{args.plot_format}")
        stl = f"Randomly Generated Waveforms, Epoch: {i_epoch+1}"

        plot_waves_1C(
            sdat_all,
            x_g,
            i_vg,
            args,
            t_max=args.time_delta * args.discriminator_size,
            show_fig=False,
            fig_file=fig_file,
            stitle=stl,
        )

        # back to train mode
        G.train()

        mlflow.log_artifacts(f"{dirs['training_dir']}", f"{dirs['training_dir']}")
        

        # ----------- Validation Loop --------------
        G.eval()
        D.eval()
        for i_batch in range(n_val_btot):
            ### ---------- DISCRIMINATOR STEP ---------------

            # 1. get real data
            # get random sample
            (data_b, ln_cb, i_vc) = sdat_val.get_rand_batch()
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
            grads = torch.cat([grads_wf, grads_cn], 1)

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
            i_vg = sdat_val.get_rand_cond_v()
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
        
        # TODO: Change back to: ... % 10 == 0
        # shorthand: ... % 4 == 2
        if (i_epoch + 1) % 10 == 0:
            save_loc_epoch = f"{dirs['output_dir']}/model_epoch_{i_epoch + 1:05}"
            mlflow.pytorch.save_model(G, save_loc_epoch)
            mlflow.pytorch.log_model(G, save_loc_epoch)

            metrics_dir = os.path.join(save_loc_epoch, "metrics")
            if not os.path.exists(metrics_dir):
                os.makedirs(metrics_dir)

            grid_dir = os.path.join(save_loc_epoch, "grid_plots")
            if not os.path.exists(grid_dir):
                os.makedirs(grid_dir)

            fig_dir = os.path.join(save_loc_epoch, "figs")
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
                
            epoch_loc_dirs = {
                "output_dir": save_loc_epoch,
                "metrics_dir": metrics_dir,
                "grid_dir": grid_dir,
                "fig_dir": fig_dir,
            }
            
            n_waveforms = 72 * 5
            evaluate_model(
                G,
                n_waveforms,
                sdat_all,
                epoch_loc_dirs,
                i_epoch,
                args
            )

    # back to train mode
    G.train()

    # make lot of losses
    iep = np.arange(1, 1 + args.epochs)
    fig_file = os.path.join(dirs['training_dir'], f"/gan_losses.{args.plot_format}")
    plt.figure(figsize=(8, 6))
    plt.plot(iep, d_wloss_ep, color="C0", label="W Distance")
    plt.plot(iep, d_total_loss_ep, color="C1", label="D Loss")
    plt.plot(iep, g_loss_ep, color="C2", label="G Loss")
    plt.legend()
    plt.ylabel("W Losses")
    plt.xlabel("Epoch")
    plt.title("Wasserstein GAN 1D, 1C")
    plt.savefig(fig_file, format=f"{args.plot_format}")
    plt.close('all')
    plt.clf()
    plt.cla()

    mlflow.log_artifact(fig_file, f"{dirs['training_dir']}")

    mlflow.pytorch.save_model(G, f"{dirs['output_dir']}/model_final")
    mlflow.pytorch.log_model(G, f"{dirs['output_dir']}/model_final")

    n_waveforms = 72 * 5

    evaluate_model(
        G,
        n_waveforms,
        sdat_all,
        dirs,
        i_epoch,
        args
    )

    try:
        train_log = f'train_log.txt'
        gan_out = os.path.join(dirs['output_dir'], train_log)
        shutil.copyfile(train_log, gan_out)
        mlflow.log_artifact(train_log, f"{dirs['output_dir']}/train_log")
    except:
        print("Failed to save training log.")


if __name__ == "__main__":
    main()