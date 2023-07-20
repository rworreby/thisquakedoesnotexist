import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 12.0


# ---- plotting functions --------------
def get_real_bin_ave(s_dat, distb, ):
    # get dataframe with attributes
    df = s_dat.df
    # print("df.shape", df.shape)
    # select bin of interest
    ix = ((distb[0] <= df['dist']) & (df['dist'] < distb[1]))
    # get normalization coefficients
    df_s = df[ix]
    # mean distace
    dist_av = df_s['dist'].mean()

    return dist_av



def get_waves_real_bin(s_dat, distbs, n_samp=256):
    # get dataframe with attributes
    df = s_dat.df
    # get waves
    wfs = s_dat.x_data
    wfs = wfs.squeeze()
    # print("df.shape", df.shape)
    # print('wfs.shape', wfs.shape)
    # select bin of interest
    ix = ((distbs[0] <= df['dist']) & (df['dist'] < distbs[1]))
    # get normalization coefficients
    df_s = df[ix]
    # get waveforms
    ws_r = wfs[ix, :]
    # mean distace
    dist_av = df_s['dist'].mean()
    # print(f"Dist mean: {df_s['dist'].mean()}")
    # print(f"Dist min: {df_s['dist'].min()}")
    # print(f"Dist max: {df_s['dist'].max()}")
    ws_r = ws_r.detach().cpu().numpy()
    Nobs = ws_r.shape[0]
    # print('# Total observations', Nobs)
    # take a sample
    # if there are too many observations take a subset
    if Nobs > n_samp:
        ii = np.arange(Nobs)
        ixc = np.random.choice(ii, size=n_samp, replace=False)
        ixc.sort()
        ws_r = ws_r[ixc, :]

    return (ws_r, dist_av)


def get_waves_model(G, grf, s_dat, distv, n_batch=256, device=None, ):
    """
    Returns synthetic waveforms
    """
    distv = s_dat.fn_dist_scale(distv)

    vc_g = distv * torch.ones(n_batch, 1, device=device)

    z = grf.sample(n_batch)
    x_g = G(z, vc_g)
    x_g = x_g.squeeze()
    x_g = x_g.detach().cpu().numpy()

    return x_g



def hist_l2_plot(ax, G, grf, s_dat, distb, nplot_samp=512, n_bins=14, device=None, fontsize=9, xlabel_show=False, ylabel_show=False, legends_txt=['Data', 'Synthetic'], color_g='C1'):
    """
    :param ax: handle for plot
    :param G: generator model
    :param grf: gaussian random field
    :param s_dat: dataset
    :param distb: [dist_min,dist_max] distance bin
    :param nplot_samp: number of samples to draw
    :param n_bins: number of bins in histogram
    :param device: torch device
    :param fontsize: fontsize for plots
    :return:
    """
    stitle = f"{distb[0]:.0f}-{distb[1]:.0f} km"
    # ----- get data --------
    ws_r, dist_av = get_waves_real_bin(s_dat, distb, n_samp=nplot_samp)
    # print("ws_r.shape", ws_r.shape)
    ws_g = get_waves_model(G, grf, s_dat, dist_av, n_batch=nplot_samp, device=device)
    # print("ws_g.shape", ws_g.shape)

    l2_r = np.sum(np.square(ws_r), axis=1)
    l2_g = np.sum(np.square(ws_g), axis=1)
    # print('l2_g.shape', l2_g.shape)
    # take range from real
    l2_min = l2_r.min()
    l2_max = l2_r.max()
    ax.hist(l2_r, bins=n_bins, range=(l2_min, l2_max), density=True, edgecolor='black', facecolor='C0', alpha=0.2,
            label='Data')
    ax.hist(l2_g, bins=n_bins, range=(l2_min, l2_max), density=True, edgecolor='black', facecolor=color_g, alpha=0.2,
            label='Synthetic')
    if stitle is not None:
        ax.set_title(stitle)
    ax.legend(legends_txt, fontsize=fontsize, loc='upper right',)
    if xlabel_show:
        ax.set_xlabel('l2 norm', fontsize=fontsize)
    if ylabel_show:
        ax.set_ylabel('Normalized Counts', fontsize=fontsize)


def time_stats_plot(ax, G, grf, s_dat, distb, nplot_samp=256, device=None, fontsize=9, xlabel_show=True, ylabel_show=False, xlim=(0.0, 40.0), ylim=(1e-4, 1.0), show_title=False, legends_txt=['Data', 'Synthetic'], color_g='C1'):
    """
    :param ax: hsandle for figure
    :param G: generator model
    :param grf: GRF object
    :param s_dat: Dataset object
    :param distb: [dist_min,dist_max] distance bin
    :param nplot_samp: Number of samples to draw
    :param device: torch device to use
    :param fontsize: font size for plots
    :return: draws plot on ax
    """
    # --- get data --
    ws_r, dist_av = get_waves_real_bin(s_dat, distb, n_samp=nplot_samp)
    # print("ws_r.shape", ws_r.shape)
    ws_g = get_waves_model(G, grf, s_dat, dist_av, n_batch=nplot_samp, device=device)
    # print("ws_g.shape", ws_g.shape)

    # --------- real envelope -----------
    ws_norm = np.abs(ws_r)
    # print('ws_norm.shape: ', ws_norm.shape)
    ws_logn = np.log(ws_norm)
    ewlog = np.mean(ws_logn, axis=0)
    estdlog = np.std(ws_logn, axis=0)
    ewave = np.exp(ewlog)
    ewlup = np.exp(ewlog + estdlog)
    ewldn = np.exp(ewlog - estdlog)

    # ----- synthetic envelope ------------
    wg_norm = np.abs(ws_g)
    # print('wg_norm.shape: ', wg_norm.shape)
    wg_logn = np.log(wg_norm)
    eglog = np.mean(wg_logn, axis=0)
    egstdlog = np.std(wg_logn, axis=0)

    egave = np.exp(eglog)
    eglup = np.exp(eglog + egstdlog)
    egldn = np.exp(eglog - egstdlog)

    # --------- plot envelopes ------
    stitle = f"{distb[0]:.0f}-{distb[1]:.0f} km"
    # time domain
    dt = s_dat.dt
    Nt = ws_r.shape[1]
    tt = dt * np.arange(0, Nt)
    ax.semilogy(tt, ewave, '-', color='C0')
    ax.fill_between(tt, ewlup, ewldn, alpha=0.2, color='C0')
    ax.semilogy(tt, egave, '-', color=color_g)
    ax.fill_between(tt, eglup, egldn, alpha=0.2, color=color_g)

    if show_title:
        ax.set_title(stitle, fontsize=fontsize)
        #ax.set_xlabel('time (sec)', fontsize=10)
    if legends_txt is not None:
        ax.legend(legends_txt, fontsize=fontsize, loc='lower right', )
    if ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if xlabel_show:
        ax.set_xlabel('time (sec)', fontsize=fontsize)
    if ylabel_show:
        ax.set_ylabel('Normalized Amplitude', fontsize=fontsize)


def fig_make_waves_1C(sfig, G, grf, s_dat, distb, nplot_samp = 20, ylim=None, color='C1', left_frac=0.3, right_frac=0.7, show_labels=True, fontsize=9, device=None):
    """
        Make plot of waves
        shape of the data has the form -> (Nobs, Nt)
        :param sfig: matplotlib subfigure handle
        :param G: generator model
        :param grf: GRF object
        :param s_dat: Dataset object
        :param distb: [dist_min,dist_max] distance bin
        :param nplot_samp: Number of samples to draw
        :param device: torch device to use
        :param fontsize: font size for plots
        :param color: Color of the plot
        :param t_max: max time for the plot in sec
        :param left_frac: fraction of left margin
        :param right_frac: fraction of right margin
        :return: show a panel with the waves

        """
    # --- get data --
    dist_av = get_real_bin_ave(s_dat, distb, )
    # print("dist_av", dist_av)
    ws_g = get_waves_model(G, grf, s_dat, dist_av, n_batch=nplot_samp, device=device)
    # print("ws_g.shape", ws_g.shape)
    dt = s_dat.dt
    Nt = ws_g.shape[1]
    Np = nplot_samp
    ws_p = ws_g[:Np, :]
    # select waves to plot
    tt = dt * np.arange(Nt)
    axs_w = sfig.subplots(Np, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0},)
    #axs_w = sfig.subplots(Np, 1)
    for ik in range(Np):
        wf = ws_p[ik,:]
        axs_w[ik].plot(tt, wf, color=color)
        if ylim is not None:
            axs_w[ik].set_ylim(ylim)
    for ax in axs_w.flat:
        ax.label_outer()
    # make title
    stitle = f"Dist = {distb[0]:.0f}-{distb[1]:.0f} km"
    if show_labels:
        sfig.suptitle(stitle, fontsize=fontsize)
        sfig.supxlabel('time (sec)', fontsize=fontsize)
    sfig.subplots_adjust(left=left_frac, right=right_frac,top=0.98, bottom=0.02)


def make_val_figs_all(G, grf, s_dat, dist_dict, fig_file, figsize=(9, 36), Ni=2, Nj=3, iD1=0, iD2=1, fontsize=9, n_samp=128, device=None):

    # make figure
    fig = plt.figure(figsize=figsize,)
    #subfigs = fig.subfigures(2, 1, hspace=0.07, )
    subfigs = fig.subfigures(4, 1, )

    # --- plot time domain stats ----
    axs0 = subfigs[0].subplots(nrows=Ni, ncols=Nj, sharex='col', sharey='row',)
    # make plots
    for idx, dist in dist_dict.items():
        time_stats_plot(axs0[idx], G, grf, s_dat, dist, nplot_samp=n_samp,fontsize=fontsize, device=device)

    #subfigs[0].set_facecolor('0.9')
    subfigs[0].suptitle('Time Domain Statistics', fontsize=fontsize)
    subfigs[0].supxlabel('time (sec)', fontsize=fontsize)
    # subfigs[0].supylabel('Normalized Amplitude')
    subfigs[0].subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.8,)

    # ----- Plot waveforms ----

    # plot waveforms for bin
    keys = list(dist_dict.keys())

    dist1 = dist_dict[keys[iD1]]
    dist2 = dist_dict[keys[iD2]]

    # --- get data ---------
    fig_make_waves_1C(subfigs[1], G, grf, s_dat, dist1, nplot_samp=8, color='C1', fontsize=fontsize, device=device)
    fig_make_waves_1C(subfigs[2], G, grf, s_dat, dist2, nplot_samp=8, color='C1', fontsize=fontsize, device=device)

    # ----- Plot histograms ----
    axs1 = subfigs[3].subplots(nrows=Ni, ncols=Nj, sharey='row')
    # make plots
    for idx, dist in dist_dict.items():
        hist_l2_plot(axs1[idx], G, grf, s_dat, dist,nplot_samp=n_samp, device=device)

    subfigs[1].suptitle('L2 Histograms', fontsize=fontsize)
    subfigs[1].supxlabel('L2 sum', fontsize=fontsize)
    subfigs[3].subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92,)
    # subfigs[1].supylabel('Normalized Counts')
    #fig.show()

    # save figure
    fig.savefig(fig_file, format='png')
    # close the figure end
    plt.close()

# def spec_stats_multitaper(ws, dt=0.04, tt_band=3.0, nn_tapers=5,):
#     """
#     Calculate 1C spectra of ws array
#     :param ws: array of shape (Nobs, Nt, )
#     :param dt: sampling time
#     :param tt_band: time_bandwidth for multitaper
#     :param nn_tapers: number_of_tapers for multitaper
#     :return: ( rf, x_freq )
#              spectra rf of shape (Nobs (Nt//2+1) ), and frequency range
#     """
#     # number of observations in signal
#     Nobs = ws.shape[0]
#     # number of samples in signal
#     Nt = ws.shape[1]

#     # allocate array to store spectra of each signal
#     F_ws = np.zeros((Nobs, Nt // 2 + 1))
#     # calculate spectra using multitaper
#     for ni in range(Nobs):
#         # get time series in acceleration units
#         wt = ws[ni, :]
#         # Multitaper
#         Cspec, x_freq = mtspec(wt[:], delta=dt,
#             time_bandwidth=tt_band, number_of_tapers=nn_tapers)
#         # store result
#         F_ws[ni, :] = Cspec

#     return F_ws, x_freq



def freq_stats_plot(ax, G, grf, s_dat, distb, nplot_samp=256, xlim=(0.2, 10.0), ylim=(1e-4, 10.0), legend_txt=['Data', 'Synthetic'], device=None, fontsize=9, xlabel_show=True, ylabel_show=False, color_g='C1'):
    """
    :param ax: handle for figure
    :param G: generator model
    :param grf: GRF object
    :param s_dat: Dataset object
    :param distb: [dist_min,dist_max] distance bin
    :param nplot_samp: Number of samples to draw
    :param device: torch device to use
    :param fontsize: font size for plots
    :return: draws plot on ax
    """
    # --- get data --
    ws_r, dist_av = get_waves_real_bin(s_dat, distb, n_samp=nplot_samp)
    # print("ws_r.shape", ws_r.shape)
    ws_g = get_waves_model(G, grf, s_dat, dist_av, n_batch=nplot_samp, device=device)
    # print("ws_g.shape", ws_g.shape)
    dt = s_dat.dt
    # ----- real spectra -------
    # TODO: Needs fix for non-working spec_stats_multitaper
    Cr_f, xr_freq = spec_stats_multitaper(ws_r, dt, )
    # log space stats
    Cr_log = np.log(Cr_f)
    Crlog_ave = np.mean(Cr_log, axis=0)
    # normalize all points
    Cr_ave = np.exp(Crlog_ave)
    Cnorm = Cr_ave.max()
    Cr_ave = Cr_ave / Cnorm
    print('Cr_ave.max(): ', Cr_ave.max())

    # ----- synthetic spectra ----
    # TODO: Needs fix for non-working spec_stats_multitaper
    Cg_f, xg_freq = spec_stats_multitaper(ws_g, dt,)
    # calculate stats
    Cg_log = np.log(Cg_f)
    Cglog_ave = np.mean(Cg_log, axis=0)
    # apply normalization factor
    Cg_ave = np.exp(Cglog_ave)
    Cg_ave = Cg_ave/Cnorm
    print('Cg_ave.max(): ', Cg_ave.max())

    # ---------- make plots -------
    stitle = f"Dist = {distb[0]:.0f}-{distb[1]:.0f} km"
    ax.loglog(xr_freq, Cr_ave, '-', color='C0')
    #ax.fill_between(x_freq, Cr_dn, Cr_up, alpha=0.2, color='C0')
    ax.loglog(xg_freq, Cg_ave, '-', color=color_g)
    #x.fill_between(xg_freq, Cg_dn, Cg_up, alpha=0.2, color='C1')
    if stitle is not None:
        ax.set_title(stitle)
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.legend(legend_txt, fontsize=fontsize)
    if xlabel_show:
        ax.set_xlabel('Freq (Hz)', fontsize=10)
    if ylabel_show:
        ax.set_ylabel('Normalized Amplitude', fontsize=10)


def make_stats_figs(G, grf, s_dat, dist_dict, fig_file, figsize=(12, 12), Ni=2, Nj=3, fontsize=9, n_samp=128, device=None):

    # make figure
    fig = plt.figure(figsize=figsize,)
    #subfigs = fig.subfigures(2, 1, hspace=0.07, )
    subfigs = fig.subfigures(2, 1, )
    # ---  freq domain stats ----
    axs0 = subfigs[0].subplots(nrows=Ni, ncols=Nj, sharex='col', sharey='row',)
    # make plots
    for idx, dist in dist_dict.items():
        freq_stats_plot(axs0[idx], G, grf, s_dat, dist, nplot_samp=n_samp, xlim=(0.4, 10.0), ylim=(1e-3, 10.0), color_g='C1', fontsize=fontsize, device=device)
    #subfigs[1].suptitle('Freq Domain statistics', fontsize=fontsize)
    #subfigs[1].supxlabel('Freq (Hz)', fontsize=fontsize)
    # subfigs[0].supylabel('Normalized Amplitude')
    #subfigs[1].subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.8,)


    # --- plot time domain stats ----
    axs1 = subfigs[1].subplots(nrows=Ni, ncols=Nj, sharex='col', sharey='row',)
    # make plots
    for idx, dist in dist_dict.items():
        time_stats_plot(axs1[idx], G, grf, s_dat, dist, nplot_samp=n_samp,fontsize=fontsize, device=device)
    #subfigs[0].set_facecolor('0.9')
    # subfigs[1].suptitle('Time Domain Statistics', fontsize=fontsize)
    # subfigs[1].supxlabel('time (sec)', fontsize=fontsize)
    # # subfigs[0].supylabel('Normalized Amplitude')
    # subfigs[1].subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.8,)
    # save figure
    fig.savefig(fig_file, format='png')
    # close the figure end
    plt.close()
    #plt.show()

