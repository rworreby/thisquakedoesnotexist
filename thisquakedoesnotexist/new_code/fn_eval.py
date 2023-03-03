#%%
# import libraries
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib


import numpy as np
import pickle as pkl
from imp import reload

# pytorch imports
import torch
from torch.autograd import Variable
from gan_d1 import Discriminator, Generator

from mtspec import mtspec

#%%
cuda=True
# --- utility functions ----
def rand_clip(av, Nv, v_min=0.0, v_max=0.1, std=0.1):
    vs = av + std * np.random.standard_normal(size=Nv)
    vs = np.clip(vs, v_min, v_max)
    return vs

def psave_waves_3C(ws, dt, t_max=40.0, Np=10, color='C0', ):
    """
        Make plot of waves
        shape of the data has the form -> (Nobs, 3, Nt)
        :param ws_p: numpy.ndarray of shape (Nobs, 3, Nt)
        :param Np: number of 3C waves to show
        :param color: Color of the plot
        :param t_max: max time for the plot in sec
        :param chan: channel type, default chan_first
        :return: show a panel with the waves

        """
    Nt = int(t_max / dt)
    Nobs = ws.shape[0]
    if Nobs < Np:
        # in case there are less observations than points to plot
        Np = Nobs

    # Select waveforms at random
    # get all indexes
    ix_all = np.arange(Nobs)
    # get samples
    ix_samp = np.random.choice(ix_all, size=Np, replace=False)
    ix_samp.sort()
    # select waves
    ws_p = ws[ix_samp, :, :Nt]
    # select waves to plot
    tt = dt * np.arange(Nt)
    plt.figure()
    fig, axs = plt.subplots(Np, 3, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                            figsize=(14, 10))
    for i in range(Np):
        for j in range(3):
            wf = ws_p[i, j, :]
            if Np > 1:
                axs[i, j].plot(tt, wf, color=color)
            else:
                axs[j].plot(tt, wf, color=color)

    for ax in axs.flat:
        ax.label_outer()
    # plt.show()

def get_model(filename):
    Z_SIZE = 100
    print('model file: ', filename)
    m_dict = torch.load(filename)
    # instatiate generator model
    G = Generator(z_size=Z_SIZE)
    # print(G)
    if cuda:
        # move models to GPU
        G.cuda()
    # load state dictionary
    G.load_state_dict(m_dict['state_dict'])
    return G

def noise(Nbatch, dim):
    # Generate noise from a uniform distribution
    m = 3
    return np.random.normal(size=[Nbatch, m, dim]).astype(
        dtype=np.float32)

def gen_get_waves(G, dist, mag, vs30, n_batch=128, z_size=100,):
    """
    Returns synthetic waveforms
    """

    vc_list = [dist * torch.ones(n_batch, 1).cuda(),
               mag * torch.ones(n_batch, 1).cuda(),
               vs30 * torch.ones(n_batch, 1).cuda(), ]
    z = noise(n_batch, z_size)
    z = torch.from_numpy(z).float().cuda()
    G.eval()
    wfs = G(z, *vc_list)
    # detach() to stop autogradient -> require_gradient = False
    # cpu() to move to cpu
    # numpy() read as numpy array
    wfs = wfs.detach().cpu().numpy()
    wfs = wfs[:, :, :, 0]

    return wfs

def get_cnorms_model(dd_v, m_v, vs30_v, gmm_nn, dataset, Ns = 128):
    # ---------- input parameters ----------
    lc0_sd = 0.42
    lc1_sd = 0.41
    lc2_sd = 0.40
    sd_v = 1.8
    sd_t = 0.85
    mp = 0.005
    # ----------------------------------------
    xp = np.array(
        [dataset.dist_scale(dd_v), dataset.mag_scale(m_v) , dataset.vs30_scale(vs30_v) ]).reshape((1,3))
    #print("xp shape", xp.shape)
    # move to gpu
    xp = torch.from_numpy(xp).float()
    if cuda:
        xp = xp.cuda()
    y = gmm_nn(xp)
    yp = y.data.cpu().numpy()
    # print("yp shape", yp.shape)

    lcm0 = dataset.fn_to_real(yp[0,0])
    lcm1 = dataset.fn_to_real(yp[0,1])
    lcm2 = dataset.fn_to_real(yp[0,2])

    # print(f"c0-> av: {lcm0} , std: {lc0_sd}")
    # print(f"c1-> av: {lcm1}, std {lc1_sd}")
    # print(f"c2-> av: {lcm2}, std {lc2_sd}")

    mp0 = lcm0 * mp
    lv0 = (2 * mp0 * np.random.random_sample() - mp0) + lcm0
    lc0_g = rand_clip(lv0, Ns, v_min=lv0 - sd_v * lc0_sd, v_max=lv0 + sd_v * lc0_sd, std=sd_t * lc0_sd)
    c0_g = np.power(10, lc0_g)
    tl0_g = f"lc0_g av: {lc0_g.mean()}"
    #print(tl0_g)
    mp1 = lcm1 * mp
    lv1 = 2 * mp1 * np.random.random_sample() + (lcm1 - mp1)
    lc1_g = rand_clip(lv1, Ns, v_min=lv1 - sd_v * lc1_sd, v_max=lv1 + sd_v * lc1_sd, std=sd_t * lc1_sd)
    c1_g = np.power(10, lc1_g)
    tl1_g = f"lc1_g av: {lc1_g.mean()}"
    #print(tl1_g)

    mp2 = lcm2 * mp
    lv2 = 2 * mp2 * np.random.random_sample() + (lcm2 - mp2)
    lc2_g = rand_clip(lv2, Ns, v_min=lv2 - sd_v * lc2_sd, v_max=lv2 + sd_v * lc2_sd, std=sd_t * lc2_sd)
    c2_g = np.power(10, lc2_g)
    tl2_g = f"lc2_g av: {lc2_g.mean()}"
    #print(tl2_g)

    ret_v = np.hstack((
        c0_g.reshape((-1,1)),
        c1_g.reshape((-1,1)),
        c2_g.reshape((-1,1))
    ))
    #print("ret_v shape", ret_v.shape)
    return ret_v

def get_waves_model(G, s_dat, gmm_nn, dataset, dd_v, m_v, vs30_v, ):

    print('dist={:.2f}, mag={:.2f}, vs30={:.2f}'.format(dd_v, m_v, vs30_v))
    # convert values to synthetic
    dist_n = s_dat.to_syn(dd_v, 'dist')
    mag_n = s_dat.to_syn(m_v, 'mag')
    vs30_n = s_dat.to_syn(vs30_v, 'vs30')
    ws_g = gen_get_waves(G, dist_n, mag_n, vs30_n,)
    print('ws_g shape: ', ws_g.shape)
    c_g = get_cnorms_model(dd_v, m_v, vs30_v, gmm_nn, dataset)
    # apply normalization contant [:, :, np.newaxis]
    # c_g = c_g[:, :, np.newaxis]
    # wc_g = ws_g*c_g
    # return waves and their corresponding normalization constant
    return (ws_g, c_g)

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

def spec_stat_multitaper(ws, dt=0.01, tt_band=3.0, nn_tapers=5):
    """

    :param ws: array of shape (Nobs, Nt, 3)
    :param dt: sampling time
    :param tt_band: time_bandwidth for multitaper
    :param nn_tapers: number_of_tapers for multitaper
    :return: ( rf, x_freq )
             rf of shape (Nt//2+1) spectra, and frequency range
    """
    # number of observations in signal
    Nobs = ws.shape[0]
    # number of samples in signal
    Nt = ws.shape[2]

    # allocate array to store spectra of each signal
    F_ws = np.zeros((Nobs, 3, Nt // 2 + 1))
    # calculate spectra using multitaper
    for i_n in range(Nobs):
        for i_chan in range(3):
            # Multitaper
            Cspec, x_freq = mtspec(ws[i_n, i_chan, :], delta=dt, time_bandwidth=tt_band, number_of_tapers=nn_tapers)
            # store result
            F_ws[i_n, i_chan, :] = Cspec

    # number of points in freP_g5_c5q domain
    Nf = x_freq.size
    # create array to store response
    Cspec_f = np.zeros((Nobs, Nf))

    # calculate reponse
    for i in range(Nobs):
        Cspec_f[i, :] = np.sqrt((F_ws[i, 0, :] ** 2 + F_ws[i, 1, :] ** 2 + F_ws[i, 2, :] ** 2))

    return Cspec_f, x_freq
    # # take an average
    # C_ave = np.mean(Cspec_f, axis=0)
    # C_std = np.std(Cs55151pec_f, axis=0)
    # return (C_ave, x_freq, C_std)

def spec_3vn_multitaper(ws, dt=0.05, tt_band=3.0, nn_tapers=5, acc_out=True,chan_norm=False):
    """
    Calculate 3C spectra of ws array
    :param ws: array of shape (Nobs, Nt, 3)
    :param dt: sampling time
    :param tt_band: time_bandwidth for multitaper
    :param nn_tapers: number_of_tapers for multitaper
    :param acc_out: output acceleration units m/s^2
    :param chan_norm: normalization by channel
    :return: ( rf, x_freq )
             spectra rf of shape (Nobs (Nt//2+1) ), and frequency range
    """
    # number of observations in signal
    Nobs = ws.shape[0]
    # number of samples in signal
    Nt = ws.shape[2]
    # get all normalization constants
    Cns_v = np.max(np.abs(ws), axis=2)
    Cns = np.zeros((Nobs),)
    for ki in range(Nobs):
        Cns[ki] = np.sqrt((Cns_v[ki, 0] ** 2 + Cns_v[ki, 1] ** 2 + Cns_v[ki, 2] ** 2))

    if chan_norm==True:
        ws_norm = np.max(np.abs(ws), axis=2)
        # for braadcasting
        ws_norm = ws_norm[:, :, np.newaxis]
        # apply normalization
        ws = ws / ws_norm

    # allocate array to store spectra of each signal
    F_ws = np.zeros((Nobs, 3, Nt // 2 + 1))

    # calculate spectra using multitaper
    for ni in range(Nobs):
        for i_chan in range(3):
            # get time series in acceleration units
            wt = ws[ni, i_chan, :]
            # Multitaper
            Cspec, x_freq = mtspec(wt[:], delta=dt,
                time_bandwidth=tt_band, number_of_tapers=nn_tapers)
            # store result
            F_ws[ni, i_chan, :] = Cspec

    # number of points in freq domain
    Nf = x_freq.size
    # create array to store response
    Cspec_f = np.zeros((Nobs, Nf))
    # calculate 3C
    for ni in range(Nobs):
        Cspec_f[ni, :] = np.sqrt((F_ws[ni, 0, :] ** 2 + F_ws[ni, 1, :] ** 2 + F_ws[ni, 2, :] ** 2))
        if acc_out==True:
            Cfn = np.max(Cspec_f[ni, :])
            Cspec_f[ni, :] = Cns[ni]*Cspec_f[ni, :]/Cfn

    # return result
    return Cspec_f, x_freq



def spec_cn_multitaper(ws, dt=0.05, tt_band=3.0, nn_tapers=5, acc_out=True,chan_norm=False):
    """
    Calculate 3C spectra of ws array
    :param ws: array of shape (Nobs, Nt, 3)
    :param dt: sampling time
    :param tt_band: time_bandwidth for multitaper
    :param nn_tapers: number_of_tapers for multitaper
    :param acc_out: output acceleration units m/s^2
    :return: ( rf, x_freq )
             spectra rf of shape (Nobs (Nt//2+1) ), and frequency range
    """
    # number of observations in signal
    Nobs = ws.shape[0]
    # number of samples in signal
    Nt = ws.shape[2]
    if chan_norm==True:
        ws_norm = np.max(np.abs(ws), axis=2)
        # for braadcasting
        ws_norm = ws_norm[:, :, np.newaxis]
        # apply normalization
        ws = ws / ws_norm

    # allocate array to store spectra of each signal
    F_ws = np.zeros((Nobs, 3, Nt // 2 + 1))
    # calculate spectra using multitaper
    for ni in range(Nobs):
        for i_chan in range(3):
            # get time series in acceleration units
            wt = ws[ni, i_chan, :]
            # time domain normalization constant
            Cnt = np.max(np.abs(wt[:]))
            # Multitaper
            Cspec, x_freq = mtspec(wt[:], delta=dt,
                time_bandwidth=tt_band, number_of_tapers=nn_tapers)
            # freq normalization
            Cnf = Cspec.max()
            # normalize spectrum to acceleration
            if acc_out==True:
                #print("Applying normalization ...")
                Cspec = Cnt*Cspec/Cnf
            # store result
            F_ws[ni, i_chan, :] = Cspec

    # number of points in freq domain
    Nf = x_freq.size
    # create array to store response
    Cspec_f = np.zeros((Nobs, Nf))
    # calculate 3C
    for ni in range(Nobs):
        Cspec_f[ni, :] = np.sqrt((F_ws[ni, 0, :] ** 2 + F_ws[ni, 1, :] ** 2 + F_ws[ni, 2, :] ** 2))
    # return result
    return Cspec_f, x_freq

def get_idxs_tp(fs,tp_fq):
    """
     get get intexes corresponding to the tuple
    :param fs: 1-D evenly spaced vector
    :param tp_fq: Tuple (fq_min, fq_max)
    :return: (i_min, i_max) Tuple with integer indexes
    """
    fq_min, fq_max = tp_fq
    fs0 = fs[0]
    dfs = fs[1]-fs[0]
    i_min = np.floor( (fq_min-fs0)/dfs ).astype(int)
    i_max = np.floor( (fq_max-fs0)/dfs ).astype(int)
    return (i_min, i_max)


#%%