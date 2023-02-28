#%%
import numpy as np
import pandas as pd

DATA_FILE = '/scratch/mflorez/gmpw/train_ffn1D/downsamp_5x_all.npy'
ATTR_FILE = '/scratch/mflorez/gmpw/train_ffn1D/wforms_table.csv'

# load dataa
print('Loading data ...')
wfs = np.load(DATA_FILE)
print('Loaded samples: ', wfs.shape[0])
# total number of training samples
Nsamp = wfs.shape[0]

# load attributes
df = pd.read_csv(ATTR_FILE)
print('wfs shape ', wfs.shape)
print('Shape attribute file: ', df.shape)

#%%
# absolute value
wfs_p = np.abs(wfs)
# calculate the max amplitude for each trace
wa = np.max(wfs_p, axis=2)
pga_v = np.max(wa, axis=1)
print('pga_v', pga_v.shape)

# store PGA vector
df['pga'] = pga_v[:]
# save results
df.to_csv(ATTR_FILE, index=False)
#
# quick test
ix = 1900
a = wfs_p[ix, :, :].max()

print('Test Result: ', a==pga_v[ix])
print('Done!')


#%%
