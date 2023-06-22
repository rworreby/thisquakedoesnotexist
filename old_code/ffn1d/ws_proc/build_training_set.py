# %%
import numpy as np
import h5py
import pandas as pd
# Params
input_data = '/scratch/mflorez/gmpw/downsamp_5x_all.h5'
outfile = '/scratch/mflorez/gmpw/train_ffn1D/downsamp_5x_all'
# initial time oaf recording
time_ini_sec = 4.0
# length of recording
tt_wave_sec = 50.0
sample_rate = 20.0
# read adataset
print('Loading ', input_data, ' ...')
f = h5py.File(input_data,'r')
d_meta = f['numMeta']
d_wf = f['wforms']
d_fname = f['wformFullNames']
print('d_wf shape: ', d_wf.shape)

# %%
# Load all waveforms
wfs = d_wf[:,:,:]
# slice data
# start time
itt_0 = int(time_ini_sec * sample_rate)
# end time
itt_end = int(tt_wave_sec * sample_rate + itt_0)
wfs = wfs[:, itt_0:itt_end, :]
wfs = wfs.transpose((2, 0, 1))

np.save(outfile, wfs)
print('out wfs shape: ', wfs.shape)
print('Done!')
# #%%
# # ------- Build Attribute file -----
# attr_file = '/Users/manuel/seisML/raw_data/wforms_table.csv'
# outfile = '/Users/manuel/seisML/raw_data/mini_wforms_table.csv'
# df_cat = pd.read_csv(attr_file)
#
# df = df_cat.iloc[:31000,:]
# df.to_csv(outfile,index=False)
#%%
