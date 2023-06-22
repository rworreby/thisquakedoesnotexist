# %%

# -------- QC Training set -------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# from mtspec import mtspec

# ---------------------- INPUT PARAMETERS -----------------------
config_d = {
    # directory layout
    'data_file': '/scratch/mflorez/gmpw/train_ffn1D/downsamp_5x_all.npy',
    'attr_file': '/scratch/mflorez/gmpw/train_ffn1D/wforms_table.csv',
    'vs30_file': '/scratch/mflorez/gmpw/train_ffn1D/japan_vs30.csv',
    'out_dir': '/scratch/mflorez/gmpw/train_ffn1D/train_Dist/',
    # selection criteriaa
    'MAX_PGA': None,
    'MIN_MAG': 5.2,
    'MAX_MAG': 6.2,
    'MIN_DIST': 40.0,
    'MAX_DIST': 180.0,
    'MIN_DEP': 0.0,
    'MAX_DEP': 60.0,
    'Vs30_MIN': 0.1,
    'Vs30_MAX': 0.9,
    # min number of observations per stations
    'min_obs': 20,

}


# ----------------------------------------

def init_conf(conf_d):
    """
    Output configurations dictionary and create data directory
    """
    out_dir = conf_d['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = out_dir + 'config_d.json'
    # write out config file
    with open(fname, 'w') as f:
        json.dump(conf_d, f, indent=4)

    return out_dir


DATA_FILE = config_d['data_file']
ATTR_FILE = config_d['attr_file']
VS30_FILE = config_d['vs30_file']
MIN_OBS = config_d['min_obs']

Vs30_MIN = config_d['Vs30_MIN']
Vs30_MAX = config_d['Vs30_MAX']

# load data
print('Loading data ...')
wfs = np.load(DATA_FILE)

# load attributes
df = pd.read_csv(ATTR_FILE)
print('Shape attribute file: ', df.shape)

# load vs30
dfvs = pd.read_csv(VS30_FILE)
print('Shape vs30 file: ', dfvs.shape)

# get stationss
print('# stations: ', len(df['sta'].unique()))
OUT_DIR = init_conf(config_d)

# ix = (7.45 < df['mag']) & (df['mag'] <= 7.6)
# df.loc[ix,['mag']] = 7.5
# print("mags ok")


# %%
# prepare vs30 dat
dfv = dfvs[['sta', 'vs30']].copy()
dfv['vs30'] = np.round(dfv['vs30'] / 1000.0, 3)
# keep only stations with a valid range of vs30
ix = (Vs30_MIN <= dfv['vs30']) & (dfv['vs30'] <= Vs30_MAX)
dfv = dfv[ix]
# plt.figure()
# plt.hist(dfv['vs30'], bins=20)
# plt.show()

# plt.figure()
# plt.hist(dfv['vs30'], bins=20)
# plt.show()

# %%
# ---- Summary statistics -----
# -> sort by stations
# group by station
dfbysta = df.groupby('sta')
res = dfbysta.count()['otime_str'].sort_values(ascending=False)
# convert res series  into data frame
df_nn = res.reset_index(name='sta_nn')
print('df_nn', df_nn.shape)
# join with original data frame
df_f = pd.merge(df, df_nn, on='sta', how='left')
# join with vs30 file
df_f = pd.merge(df_f, dfv, on='sta', how='inner')
print('Shape valid attributes: ', df_f.shape)

# %%
# select stations with more than observations
df_f = df_f[df_f['sta_nn'] > MIN_OBS]
MAX_PGA = config_d['MAX_PGA']
MIN_MAG = config_d['MIN_MAG']
MAX_MAG = config_d['MAX_MAG']
MIN_DIST = config_d['MIN_DIST']
MAX_DIST = config_d['MAX_DIST']
MIN_DEP = config_d['MIN_DEP']
MAX_DEP = config_d['MAX_DEP']

# only subduction
#df_f = df_f[df_f['eq_type'] == 'subduction']

sel_i = (MIN_MAG <= df_f['mag']) & (df_f['mag'] <= MAX_MAG) & \
        (MIN_DIST <= df_f['dist']) & (df_f['dist'] <= MAX_DIST) & \
        (MIN_DEP <= df_f['ev_dep']) & (df_f['ev_dep'] <= MAX_DEP)
# sel_i = sel_i & (df_f['sta'] == 'IWTH04')

df_f = df_f[sel_i]

# number of samples
Nobs = df_f.shape[0]
print('df_f  selected attributes shape', df_f.shape)
print('Number of waves: ', Nobs)
print('Min Magnitude: ', df_f['mag'].min())
print('Max Magnitude: ', df_f['mag'].max())
print('Min Dist: ', df_f['dist'].min())
print('Max Dist: ', df_f['dist'].max())
print('Min Dep: ', df_f['ev_dep'].min())
print('Max Dep: ', df_f['ev_dep'].max())

print('# Selected stations')
print(len(df_f['sta'].unique()))

# %%
# get relevant waveforms
# get indeces
i_wf = np.array(df_f['i_wf'])
wfs = wfs[i_wf, :, :]
print('wfs shape: ',wfs.shape)
# reset waveform indeces in dataframe
df_f['i_wf'] = np.arange(Nobs)
# finally reset the index
df_f = df_f.reset_index(drop=True)
# get numpy array with attributes
df_f = df_f[['mag','dist','ev_dep','pga','vs30']]
vc = df_f.to_numpy()
print("vc shape", vc.shape)


# save the resusssslts
out_attr = OUT_DIR + 'wforms_table_sel.csv'
out_wfile = OUT_DIR + 'downsamp_5x_sel'
df_f.to_csv(out_attr, index=False)
np.save(out_wfile, wfs)


print('Done!')

#
# # save the results
# out_attr = OUT_DIR + 'vc_sel'
# out_wfile = OUT_DIR + 'downsamp_5x_sel'
# #df_f.to_csv(out_attr, index=False)
# np.save(out_attr, vc)
# np.save(out_wfile, wfs)
# print('Done!')

# %%
