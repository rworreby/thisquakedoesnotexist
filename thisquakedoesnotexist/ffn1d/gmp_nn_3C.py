#%%
# ------------------- BASE NN SAMPLE --------------------
##
import os
#import helpers
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
#import helpers ddss

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
#sns.set(style="white", color_codes=True)
#sns.set(color_codes=True)

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
# import libraries
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 10.0

CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


# ----- Input Parameters ------
config_d = {
    'data_file': '/scratch/mflorez/gmpw/trainP_1x_set/train_P45_vs30/downsamp_1x_sel.npy',
    'attr_file': '/scratch/mflorez/gmpw/trainP_1x_set/train_P45_vs30/wforms_table_sel.csv',
 }
ATTR_FILE = config_d['attr_file']
DATA_FILE = config_d['data_file']

#%%
# factory function to rescale variables
def make_maps_scale(v_min, v_max):
    """
    Rescale arrays based on max_v, where min comes from the data
    """

    def to_scale_11(x):
        dv = v_max-v_min
        xn = (x-v_min) / dv
        # rescale to [-1,1]
        xn = 2.0*xn -1.0
        return xn
    def to_real(x):
        dv = v_max-v_min
        xr = (x+1.0)/2.0
        xr = xr*dv+v_min
        return xr
    # return tuple of functions
    return (to_scale_11, to_real)

def make_rescale(v):
    """
    Reascale numpy array to be in the range 0,1
    """
    v_min = v.min()
    v_max = v.max()
    def rescale(x):
        dv = v_max-v_min
        # rescale to [0,1]
        xn = (x-v_min) / dv
        return xn
    # rescale to [-1,1]
    #vn = 2.0 * vn - 1.0
    return rescale


class PGADataset(Dataset):
    def __init__(self, data_file, attr_file, PGA_MAX=None):
        # Load all data
        df = pd.read_csv(attr_file)
        print('Shape attribute file: ', df.shape)
        # load data
        print('Loading waveforms ...')
        wfs = np.load(data_file)
        print('Loaded waveforms: ', wfs.shape[0])

        # -------------- set input variables ----------------
        df = df[['mag','dist','ev_dep','pga','vs30','sta','otime_str']]
        self.df = df
        # create scaling functions
        dist = np.array(df['dist']).reshape(-1,1)
        mag = np.array(df['mag']).reshape(-1,1)
        vs30 = np.array(df['vs30']).reshape(-1,1)
        pga = np.array(df['pga']).reshape(-1,1)

        self.dist_scale = make_rescale(dist)
        self.mag_scale = make_rescale(mag)
        self.vs30_scale = make_rescale(vs30)

        # create array for conditional variables
        distv = self.dist_scale(dist)
        magv = self.mag_scale(mag)
        vs30v = self.vs30_scale(vs30)
        # create conditional variables
        vs = np.concatenate( (distv, magv, vs30v), axis=1)
        print('vs[:,0] min: ',vs[:,0].min())
        print('vs[:,1] min: ',vs[:,1].min())
        print('vs[:,2] min: ',vs[:,2].min())

        # -------------- set outputl/abels variables ----------------
        print('calculating normalizations ...')
        wfs_norm = np.max(np.abs(wfs), axis=2)
        # store normalization facots
        cnorms = wfs_norm.copy()
        self.cnorms = cnorms
        # --- rescale norms -----
        vns_max = np.max(cnorms, axis=1)
        vns_min = np.min(cnorms, axis=1)
        pga_max = vns_max.max()
        pga_min = vns_min.min()
        log_pga_max = np.log10( pga_max )
        log_pga_min = np.log10( pga_min )
        lc_m = np.log10(cnorms)
        if PGA_MAX is not None:
            fn_to_scale_11, fn_to_real = make_maps_scale( log_pga_min, np.log10(PGA_MAX))
        else:
            fn_to_scale_11, fn_to_real = make_maps_scale( log_pga_min, log_pga_max)

        # create array for rescaled varibles
        ln_cns = np.zeros_like(lc_m)
        ln_cns[:,0] = fn_to_scale_11(lc_m[:,0])
        ln_cns[:,1] = fn_to_scale_11(lc_m[:,1])
        ln_cns[:,2] = fn_to_scale_11(lc_m[:,2])
        # store variables of interest
        self.fn_to_scale_11 = fn_to_scale_11
        self.fn_to_real = fn_to_real
        self.ln_cns = ln_cns

        # convert to tensors
        self.vs = torch.from_numpy(vs).float()
        self.y_l = torch.from_numpy(ln_cns).float()
        print('self.vs shape: ', self.vs.shape)
        print('self.y_l shape: ', self.y_l.shape)


    def __len__(self):
        return self.vs.size(0)

    def __getitem__(self, idx):
        return ( self.vs[idx,:], self.y_l[idx, :] )

# # ------- uncomment for data loader tests
# # PGA_MAX = 9.8 * 0.05 242378
# dataset = PGADataset(data_file=DATA_FILE, attr_file=ATTR_FILE )
# train_dataset, val_dataset = random_split(dataset, [90000, 16189])
# train_loader = DataLoader(dataset=train_dataset, batch_size=128)
# dat, y_l = next(iter(train_loader))
# print('shapes: ', dat.shape, y_l.shape)
#
# # plot histograms
# iC=2
# ln_pga = np.log10(dataset.cnorms[:,iC])
# pga_n = dataset.fn_to_scale_11(ln_pga)
# pga_r = dataset.fn_to_real(pga_n)
#
# plt.figure()
# plt.hist(ln_pga, bins=80, color='C0')
# plt.xlabel('log(PGA) $(m/s^2)$')
# plt.ylabel('cnts')
# plt.title('REAL INIT')
# plt.show()
#
# plt.figure()
# plt.hist(pga_n, bins=80, color='C1')
# plt.xlabel('log(PGA) $(m/s^2)$')
# plt.ylabel('cnts')
# plt.title('NORMED')
# plt.show()
#
#
# plt.figure()
# plt.hist(pga_r, bins=80, color='C2')
# plt.xlabel('log(PGA) $(m/s^2)$')
# plt.ylabel('cnts')
# plt.title('TO REAL')
# plt.show()
# #
# mc = dataset.cnorms[:,:]
# print("mc", mc.max())

#%%

# define the NN architecture
class Net(nn.Module):
    def __init__(self, n_vs=3, hidden_1=64, hidden_2=256 ):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            # linear layer (n_vs -> hidden_1)
            nn.Linear(n_vs, hidden_1), torch.nn.ReLU(),
            # linear layer (hidden_1 -> hidden_2)
            nn.Linear(hidden_1, hidden_2), torch.nn.ReLU(),
            # linear layer (hidden_2 -> hidden_2)
            nn.Linear(hidden_2, hidden_2), torch.nn.ReLU(),
            # nn.Linear(hidden_2, 2*hidden_2), torch.nn.ReLU(),
            # nn.Linear(2*hidden_2, hidden_2), torch.nn.ReLU(),

            # linear layer (hidden_2 -> hidden_1)
            nn.Linear(hidden_2, hidden_1), torch.nn.ReLU(),
            # linear layer (hidden_2 -> 1)
            nn.Linear(hidden_1, 3),
            # tanh output
            torch.nn.Tanh()
        )
    def forward(self, x):
        # input shape
        # print('shape x init:', x.shape)
        x = self.layers(x)
        return x

# takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    # for every Linear layer in a model..
    if type(m) == nn.Linear:
        # get the number of the inputs
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)

# ---- Loss function  -----
def loss_fn(y_m, y_l):
    dloss = torch.mean((y_m-y_l)**2)
    return dloss


# # -- uncomment for testing --
# # ------- Load Model ----------
# model = Net()
# # -> train on GPU if available
# if CUDA:
#     # move models to GPU
#     model.cuda()
#     print('GPU available for training. Models moved to GPU')
# else:
#     print('Training on CPU.')
# print(model)
# dataset = PGADataset(data_file=DATA_FILE, attr_file=ATTR_FILE )
# train_dataset, val_dataset = random_split(dataset, [90000, 16189])
# train_loader = DataLoader(dataset=train_dataset, batch_size=128)
# val_loader = DataLoader(dataset=val_dataset, batch_size=128)
# dat, y_l = next(iter(train_loader))
# if CUDA:
#     dat = dat.cuda()
#     y_l = y_l.cuda()
#
# output = model(dat)
# print('output shape ', output.shape)
# loss_v = loss_fn(output,y_l)
# print('loss fn: ', loss_v.item())

# %%
n_epochs = 80
lr = 1e-5
BATCH_SIZE = 256
# Training loss
criterion = loss_fn

# ------- Load Model ----------
model = Net()
# ->
# -> train on GPU if available
if CUDA:
    # move models to GPU
    model.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')
# initialize weights
model.apply(weights_init_normal)

print(model)
# Load data
dataset = PGADataset(data_file=DATA_FILE, attr_file=ATTR_FILE, PGA_MAX=28.0 )
train_dataset, val_dataset = random_split(dataset, [90000, 16189])
train_loader = DataLoader(dataset=train_dataset, batch_size=128)
val_loader = DataLoader(dataset=val_dataset, batch_size=128)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr)


# Measurements used for graphing loss demo line
loss_batch = []
loss_cur=0.0
for epoch in range(1, n_epochs+1):
    # initialize var to monitor training loss
    if (epoch < 60 or loss_cur > 0.025):
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        print(f"--- epoch: {n_epochs} ---")
        for dat, y_l in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # move data to GPU
            if CUDA:
                dat = dat.cuda()
                y_l = y_l.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(dat)
            # calculate the batch loss
            loss = criterion(output, y_l)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record average batch loss
            loss_cur = loss.item()
            loss_batch.append(loss_cur)

# plot loss
#n_plot = 1500*10
plt.plot(np.array(loss_batch))
plt.title('Loss')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.show()
print(f"{epoch}: {loss_cur}")
# %%
#save generator
torch.save({'state_dict': model.state_dict()}, f"models_3C/model_G_ep_{n_epochs}.pth")
print('best model', n_epochs,' saved')

#%%
model_file = 'models_3C/model_G_ep_80.pth'
print('model file: ', model_file)
m_dict = torch.load(model_file)
# ------- Load Model ----------
model = Net()
# ->
# -> train on GPU if available
if CUDA:
    # move models to GPU
    model.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')

model.load_state_dict(m_dict['state_dict'])
# Load data
# Load data
dataset = PGADataset(data_file=DATA_FILE, attr_file=ATTR_FILE, PGA_MAX=28.0 )
train_dataset, val_dataset = random_split(dataset, [90000, 16189])
train_loader = DataLoader(dataset=train_dataset, batch_size=256)
val_loader = DataLoader(dataset=val_dataset, batch_size=256)


# %%

# Validation results
# after training for 2 epochs, check validation accuracy dd
model = model.eval()
tol = 0.2
correct = 0
tot = 0
Ns_tot = 0
rmse = 0.0
for dat, y_l in val_loader:
    N_b = dat.size(0)
    if CUDA:
        dat = dat.cuda()
        y_l = y_l.cuda()
    output = model(dat)
    # RMS Error
    rmse += N_b*torch.sqrt(torch.mean((output.data-y_l)**2))
    # accuracy score
    tol_v = torch.abs(output.data-y_l)
    correct += (tol_v < tol).sum()
    Ns_tot += N_b

rms = rmse/Ns_tot
val_acc = 100*correct/Ns_tot
print('RMS Error: ', rms.item())
print('accuracy  ', val_acc.item(), ' %')


#%%
# folder for figure output
OUT_DIR = '/home/mflorez/seisML/figs_paper/'

DIST_MIN = 5.0
DIST_MAX = 180.0

Npts = 100
# # Evaluate the model

mag_list = [5.5, 6.0, 6.5, 7.0, 7.5]
vs0_30 = 0.3
fig_name = 'soft_soil_log'
base_txt = ''
if vs0_30 < 0.5:
    base_txt = 'Soft Soil '
else:
    base_txt = 'Bed Rock '

#stitle = ' Bed Rock Vs30 = '+ str(vs0_30)
stitle = f"{base_txt} Vs30 = {1000.0*vs0_30:.0f}"

# numpy arrays for distance and vs30
vs_30v = vs0_30*np.ones((Npts,1))
dd_p = np.linspace(start=DIST_MIN, stop=DIST_MAX, num=Npts).reshape((-1,1))

# make prediction
model = model.eval()
plt.figure(figsize=(5,4))

for mag0 in mag_list:
    magv = mag0*np.ones((Npts,1))

    xp = np.concatenate(
        (dataset.dist_scale(dd_p),
         dataset.mag_scale(magv),
         dataset.vs30_scale(vs_30v)),
        axis=1)
    # move to gpu
    xp = torch.from_numpy(xp).float()
    if CUDA:
        xp = xp.cuda()

    y = model(xp)
    yp = y.data.cpu().numpy()
    print("yp shape", yp.shape)
    txt = 'M ' + str(mag0)
    y_v = yp[:,:2]
    ypga = np.max(y_v, axis=1)
    print("ypga shape", ypga.shape)
    y0 = dataset.fn_to_real(ypga)
    y0_r = np.power(10.0,y0)
    plt.loglog( dd_p, y0_r, label=txt)
    #plt.plot( dd_p, y0, label=txt)
    if mag0 == 7.0:
        print(f"Max PGA {vs0_30}:", y0[20] )

plt.xlim((6.0, 180.0))
plt.ylim((0.02, 20.0))

plt.legend(fontsize=10)
plt.xlabel(r"Distance [km]", fontsize=10)
plt.ylabel(r"$PGA [m/s^2]$", fontsize=10)
plt.title(stitle, fontsize=10)
plt.tight_layout()
plt.grid(which='both')
fig_file = OUT_DIR + fig_name + '.pdf'
plt.savefig(fig_file,format='pdf')
plt.show()


#%%
DIST_MIN = 5.0
DIST_MAX = 180.0
iC=0
df = dataset.df
#mbs = [4.9, 5.1]
# plot file
fig_name = 'pga_D_log'
# market size
MSIZE = 1.0
mbs = [6.4, 6.6]
#mbs = [5.4, 5.6]
zb = [0.0, 150.0]
#vsb = [0.740, 0.860]
vsb = [0.700, 0.900]
#vsb = [0.280, 0.320]
#vsb = [0.250, 0.350]
ix = ((mbs[0] <= df['mag'] ) & (df['mag']< mbs[1]) &
         (vsb[0] <= df['vs30'] ) & (df['vs30']< vsb[1]) &
         (zb[0] <= df['ev_dep'] ) & (df['ev_dep']<= zb[1]) )
df_s = df[ix]
Nobs = ix.sum()
vs30_str = f"Vs30: {1000.0*vsb[0]:.0f} - {1000.0*vsb[1]:.0f} m/s"
stitle = r' '+str(mbs[0])+ ' - ' + str(mbs[1])+' M | '+vs30_str+' | $N_{obs}$ = '+ str(Nobs)

print(stitle)
print('# observations', Nobs)
cns = dataset.cnorms[ix,:]
print('shape: cns', cns.shape)
Npts = 100
# # Evaluate the model
mag0 = (mbs[0] + mbs[1])/2.0
vs0_30 = (vsb[0] + vsb[1])/2.0

magv = mag0*np.ones((Npts,1))
vs_30v = vs0_30*np.ones((Npts,1))
dd_p = np.linspace(start=DIST_MIN, stop=DIST_MAX, num=Npts).reshape((-1,1))
xp = np.concatenate(
    (dataset.dist_scale(dd_p), dataset.mag_scale(magv) ,dataset.vs30_scale(vs_30v)),
    axis=1)

# make prediction
model = model.eval()
# move to gpu
xp = torch.from_numpy(xp).float()
if CUDA:
    xp = xp.cuda()

# syn
yp_v = model(xp)
yp_v = yp_v.data.cpu().numpy()
print("yp shape", yp_v.shape)
ypga = np.max(yp_v[:,:2], axis=1)
print("ypga shape", ypga.shape)

yp = dataset.fn_to_real(ypga)
yp_r = np.power(10.0,yp)
# real
dist = df_s['dist'].to_numpy()
pga_d = np.max(cns[:,:2], axis=1)
#ln_pga = np.log10(pga_d)
# make plot
plt.figure(figsize=(5,4))
plt.loglog(dist, pga_d, '.', markersize=3.0, c='C0', alpha=0.6, label= 'Real')
plt.loglog( dd_p, yp_r, color='C1', label='Synthetic')
#plt.ylim((-2.0, 2.0))
plt.xlim((6.0, 180.0))
plt.ylim((0.009, 8.0))
plt.legend(fontsize=10)
plt.xlabel(r"Distance [km]", fontsize=10)
plt.ylabel(r"$PGA [m/s^2]$", fontsize=10)
plt.title(stitle, fontsize=10)
plt.tight_layout()
plt.grid(which='both')
fig_file = OUT_DIR + fig_name + '.pdf'
plt.savefig(fig_file,format='pdf')
plt.show()

#%%