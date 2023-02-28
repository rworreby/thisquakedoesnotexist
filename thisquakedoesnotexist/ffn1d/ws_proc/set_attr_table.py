import h5py
import numpy as np
import pandas as pd


# -------------- Input Parameters  ---------------------

OUTFILE = '/scratch/mflorez/gmpw/train_ffn1D/wforms_table.csv'

# load HDF5 data
f = h5py.File('/scratch/mflorez/gmpw/downsamp_5x_all.h5', 'r')

# --------------------------------------------

# get datasets

d_meta = f['numMeta']
d_wf = f['wforms']
d_fname = f['wformFullNames']


print('shapes: d_meta, d_wf, d_fname')
print(d_meta.shape)
print(d_wf.shape)
print(d_fname.shape)

# convert meta data to numpy array
meta_v = np.array(d_meta)
#array with file names
d_fname_v=np.array(d_fname)
# number of observations
Nobs = meta_v.shape[1]
# create dataframe with waveform attibutes
df = pd.DataFrame({
    'i_wf':     np.arange(Nobs), # index for waveform
    'mag':      meta_v[0,:],    # Event magnitude
    'dist':     meta_v[1,:],    # Hypocentral distance [km]
    'ev_dep':   meta_v[2,:],    # Earthquake hypocenter depth  [km]
    'snr':      meta_v[3,:],    # Signal/noise power ratio
    'sta_lat':  meta_v[4,:],    # Station latitude
    'sta_lon':  meta_v[5,:],    # Station longitude
    'ev_lat':   meta_v[6,:],    # Earthquake latitude
    'ev_lon':   meta_v[7,:],    # Earthquake longitude
    'back_az':  meta_v[8,:],    # back-azimuth
    'otime':    meta_v[9,:],    # Earthquake origin time
    'u_ID':     meta_v[10,:],   # Unique record ID
    'ev_ID':    meta_v[11,:],   # Unique earthquake ID
    'i_onset':  meta_v[12,:],   # Onset index
    'wf_fname': np.array(d_fname), # full filename of waveform

})
# convert  bites object to string
df['wf_fname'] = df['wf_fname'].apply(bytes.decode)

# get a pandas series with the filenames
ds_wf = df['wf_fname']
# get pandas dataframe with the contents of the path
df_wf = ds_wf.str.split('/', expand=True)
# store attributes of interest
df['region'] = df_wf[5]
df['eq_type'] = df_wf[7]
df['otime_str'] = df_wf[10]
df['net'] = df_wf[11]
# pandas series with station definitions
s_sta = df_wf[12].str.split('.').str[0]
# get the channel
df['sta_chan'] = df_wf[12].str.split('.').str[1]
# get full code for stations
df['sta_all'] = s_sta
df['sta'] = s_sta.str[:6]
df['sta_otime'] = s_sta.str[6:]
# print test if in fact only station codes were extracted
s_ot = df['otime_str'].str[2:-2]
s_so = df['sta_otime']
a = s_so == s_ot
print('Comparison with origin times')
print(a.unique())

# Analyze knt
df_net = df[df['net']=='knt']
print('k-net number of strong motion sensors: ',
      len(df_net['sta'].unique()))
print('k-net number of seismograms: ',df_net.shape[0])
# Analyze kik
df_net = df[df['net']=='kik']
print('kik-net number of strong motion sensors: ',
      len(df_net['sta'].unique()))
print('kik-net number of seismograms: ',df_net.shape[0])

# count events in dataset
n_evs = len(df['otime_str'].unique())
print('Total number of events: ', n_evs)

# save data frames
print("Attribute table shape: ", df.shape)
df.to_csv(OUTFILE, index=False)
print('---------------')
print('Done!')
