# %%
import os
import json
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pprint
#from scipy.io import savemat

# import librariesss

import torch
import torch.nn as nn
import torch.nn.functional as F
from imp import reload

from utils import *
from random_fields import *
from torch.utils.data import DataLoader
from torch import nn
from dataUtils import *
from fn_plot import *
from mtspec import mtspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 10.0

# Directory for experiments

#BASE_DIR = '/scratch/mflorez/fourier/ffn1D/exp_4x/cond_c5'
OUT_FIG_DIR = '/scratch/mflorez/fourier/ffn1D/exp_4x/figs_paper'

BASE_DIR = '/scratch/mflorez/fourier/exp_gan1d_4x/b10_c5_lr4'

# number of synthetic plots to generate
NPLT_MAX = 20
fconf = os.path.join(BASE_DIR, 'config_d.json')
with open(fconf) as json_file:
    config_d = json.load(json_file)

    # Print the type of data variable
    pprint.pprint(config_d)

# store variables of interest
DATA_FILE = config_d['data_file']
ATTR_FILE = config_d['attr_file']
# Batch size
NBATCH = config_d['batch_size']
# epochs for training
NUM_EPOCHS = config_d['epochs']

#+ -------- Generator params ------
# Parameter for gaussian random field
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

# torch device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# setup folders to store experiments
def init_dir_conf(conf_d):
    """
    Output configurations dictionary and create stats dirs
    """
    out_dir = conf_d['out_dir']
    models_dir = os.path.join(out_dir, 'models')
    # create directory for figures
    fig_dir = os.path.join(out_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    # create dir for stats
    tt_stats_dir = os.path.join(out_dir, 'stats2')
    if not os.path.exists(tt_stats_dir):
        os.makedirs(tt_stats_dir)

    return (models_dir, fig_dir, tt_stats_dir)


# Get paths
MODEL_DIR, FIG_DIR, TT_STATS_DIR = init_dir_conf(config_d)
print(TT_STATS_DIR)


#%%
dataset = WaveDatasetDist(data_file=DATA_FILE, attr_file=ATTR_FILE, ndist_bins=NDIST_BINS, dt=DT )
# create train loader
train_loader = DataLoader(dataset, batch_size=NBATCH, shuffle=True)

Ntrain = len(dataset)

# ----- test Loader-----
x_r, y_r = next(iter(train_loader))
print('shapes x_r, y_r: ', x_r.shape, y_r.shape)
Nb = x_r.size(0)


# %%
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

# ----- Load generator -----------
G = Generator(z_size=NOISE_DIM).to(device)
print(G)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number Generator parameters: ", nn_params)
# model file
EPi = 81
fmodel = os.path.join(MODEL_DIR, 'model_G_epoch_' + str(EPi) + '.pth')

m_dict = torch.load(fmodel)
G.load_state_dict(m_dict['state_dict'])


#%%


DIST_DICT= {
    (0,0): [40.00, 54.01],
    (0,1): [54.01, 68.03],
    (0,2): [68.03, 82.04],
    (1,0): [82.04, 96.05],
    (1,1): [96.05, 110.06],
    (1,2): [110.06, 124.07],
}

# --- plot time domain stats ---
Ni = 2
Nj = 3
fig, axs = plt.subplots(nrows=Ni, ncols=Nj, figsize=(7, 4))
#legends = ['Data','GANO']
legends = None
for idx, dist in DIST_DICT.items():
    time_stats_plot(axs[idx], G, grf, dataset, dist, legends_txt=legends,color_g='C1',device=device, xlabel_show=True,ylabel_show=False,)

# stitle = "Best Model CONV GAN"
# fig.suptitle(stitle, fontsize=10)
# fig.supylabel('Normalized Amplitud', fontsize=10)
# fig.supxlabel('time (sec)', fontsize=10)
fig_file = os.path.join(OUT_FIG_DIR, f"conv_time_stats_{EPi}.pdf")
fig.tight_layout()
fig.savefig(fig_file,format='pdf')
#plt.show()
#%%

# --- plot time domain stats ---
distb = DIST_DICT[(0,1)]
fig, axs = plt.subplots(figsize=(3, 2))
legends = ['Data','CONV GAN']
time_stats_plot(axs, G, grf, dataset, distb, legends_txt=legends,color_g='C1',device=device, xlabel_show=True,ylabel_show=True, show_title=False, fontsize=10)

# stitle = "CONV GAN"
# plt.title(stitle, fontsize=10)
fig_file = os.path.join(OUT_FIG_DIR, f"conv_gan_tt_main.pdf")
fig.savefig(fig_file,format='pdf')
plt.show()


#%%



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


def freq_stats_plot_std(ax, G, grf, s_dat, distb, nplot_samp=256, xlim=(0.5, 20.0), ylim=(1e-4, 10.0), device=None, fontsize=9, legends_txt=['Data', 'Synthetic'], xlabel_show=True, ylabel_show=False, show_title=False, color_g='C1'):
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
    FREQ = xlim

    # --- get data --
    ws_r, dist_av = get_waves_real_bin(s_dat, distb, n_samp=nplot_samp)
    # print("ws_r.shape", ws_r.shape)
    ws_g = get_waves_model(G, grf, s_dat, dist_av, n_batch=nplot_samp, device=device)
    # print("ws_g.shape", ws_g.shape)
    dt = s_dat.dt
    # ----- real spectra -------

    Cr_f, xr_freq = spec_stats_multitaper(ws_r, dt, )
    (i1, i2) = get_idxs_tp(xr_freq, FREQ)
    xr_freq = xr_freq[i1:i2]
    Cr_f = Cr_f[:, i1:i2]

    print(xr_freq)
    # log space stats
    Cr_log = np.log(Cr_f)
    Crlog_ave = np.mean(Cr_log, axis=0)
    Crlog_std = np.std(Cr_log, axis=0)
    # normalize all points
    Cr_ave = np.exp(Crlog_ave)
    Cnorm = Cr_ave.max()
    Cr_ave = Cr_ave / Cnorm
    print('Cr_ave.max(): ', Cr_ave.max())
    # apply std
    Cr_ln = np.log(Cr_ave)
    Cr_up = np.exp(Cr_ln + Crlog_std)
    Cr_dn = np.exp(Cr_ln - Crlog_std)


    # ----- synthetic spectra ----
    Cg_f, xg_freq = spec_stats_multitaper(ws_g, dt,)
    (i1, i2) = get_idxs_tp(xg_freq, FREQ)
    xg_freq = xg_freq[i1:i2]
    Cg_f = Cg_f[:, i1:i2]
    # calculate stats
    Cg_log = np.log(Cg_f)
    Cglog_ave = np.mean(Cg_log, axis=0)
    Cglog_std = np.std(Cg_log, axis=0)
    # apply normalization factor
    Cg_ave = np.exp(Cglog_ave)
    Cg_ave = Cg_ave/Cnorm
    print('Cg_ave.max(): ', Cg_ave.max())
    # apply std
    Cg_ln = np.log(Cg_ave)
    Cg_up = np.exp(Cg_ln + Cglog_std)
    Cg_dn = np.exp(Cg_ln - Cglog_std)


    # ---------- make plots -------
    stitle = f"{distb[0]:.0f}-{distb[1]:.0f} km"
    ax.loglog(xr_freq, Cr_ave, '-', color='C0')
    ax.fill_between(xr_freq, Cr_dn, Cr_up, alpha=0.2, color='C0')
    ax.loglog(xg_freq, Cg_ave, '-', color=color_g)
    ax.fill_between(xg_freq, Cg_dn, Cg_up, alpha=0.2, color=color_g)
    if show_title:
        ax.set_title(stitle)
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    if legends_txt is not None:
        ax.legend(legends_txt, fontsize=fontsize)
    if xlabel_show:
        ax.set_xlabel('Freq (Hz)', fontsize=10)
    if ylabel_show:
        ax.set_ylabel('Normalized Amplitude', fontsize=10)




#%%
# --- plot frequency domain stats ---


Ni = 2
Nj = 3
fig, axs = plt.subplots(nrows=Ni, ncols=Nj, figsize=(7, 4))
#legends = ['Data','GANO']
legends = None
for idx, dist in DIST_DICT.items():
    freq_stats_plot_std(axs[idx], G, grf, dataset, dist, xlim=(0.5, 12.55), ylim=(1e-3, 10.0), legends_txt=legends, color_g='C1',device=device,xlabel_show=True,ylabel_show=False)
# stitle = "GANO"
# fig.suptitle(stitle, fontsize=10)
# fig.supylabel('Normalized Amplitude', fontsize=10)
# fig.supxlabel('Freq (Hz)', fontsize=10)
fig_file = os.path.join(OUT_FIG_DIR, f"conv_freq_stats_all.pdf")
fig.tight_layout()
fig.savefig(fig_file,format='pdf')
#plt.show()



#%%

distb = DIST_DICT[(0,1)]
fig, axs = plt.subplots(figsize=(3, 2))
legends = ['Data','CONV GAN']

freq_stats_plot_std(axs, G, grf, dataset, distb, xlim=(0.5, 10.0), ylim=(1e-3, 10.0),legends_txt=legends, color_g='C1',device=device,xlabel_show=True,ylabel_show=True)
fig_file = os.path.join(OUT_FIG_DIR, f"gano_freq_stats_main.pdf")
fig.savefig(fig_file,format='pdf')
plt.show()


# %%



# --- plot histograms ---
Ni = 2
Nj = 3
fig, axs = plt.subplots(nrows=Ni, ncols=Nj, figsize=(12, 6))
for idx, dist in DIST_DICT.items():
    hist_l2_plot(axs[idx], G, grf, dataset, dist, color_g='C2',device=device)
stitle = "Best Model CONV GAN"
fig.suptitle(stitle, fontsize=10)
fig.supxlabel('l2 norm', fontsize=10)
fig.supylabel('Normalized Counts', fontsize=10)
fig_file = os.path.join(TT_STATS_DIR, f"conv_gan_l2_stats.pdf")
fig.savefig(fig_file,format='pdf')
fig.show()

#%%