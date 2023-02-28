#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = '/scratch/mflorez/gmpw/train_dp/downsamp_1x_all.npy'
ATTR_FILE = '/scratch/mflorez/gmpw/train_dp/wforms_table.csv'
OUT_DATA = '/scratch/mflorez/gmpw/train_dp/downsamp_1x_TAP_all.npy'

#%%
# Number of seconds  tapers
TSEC = 4.0
DT = 0.01

#%%
# load dataa
print('Loading data ...')
wfs = np.load(DATA_FILE)
print('Loaded samples: ', wfs.shape[0])
# total number of training samples
Nsamp = wfs.shape[0]
Nt = wfs.shape[2]
# load attributes
df = pd.read_csv(ATTR_FILE)
print('wfs shape ', wfs.shape)
print('Shape attribute file: ', df.shape)

#%%
Nt = 4000
Nsamp = 20
TSEC = 4.0
DT = 0.01

ntap = int(TSEC/DT)
print(ntap)
#get hanning window
han = np.hanning(2*ntap)
plt.plot(han)
plt.show()
#%%
# create taper
han = han[:ntap]
taper = np.hstack([han, np.ones(Nt-ntap)])

# create 3C taper
taper = np.tile(taper,(3,1))
taper = taper[np.newaxis,:,:]
# full taper
ttaper = np.tile(taper, (Nsamp,1,1))
# change the taper type
ttaper = ttaper.astype('float32')

tt = 0.01*np.arange(Nt)
plt.plot(tt,ttaper[4,2,:])
plt.show()
print("Taper shape: ", ttaper.shape)

#%%
# apply the taper to all observations
wfs = wfs*ttaper
# ssave results
np.save(OUT_DATA, wfs)
print('out wfs shape: ', wfs.shape)
print("wfs data type: ",wfs.dtype)
print('Done!')

#%%
