#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from thisquakedoesnotexist.utils.preprocessor import Preprocessor
sns.set()

def main():
    input_file = 'thisquakedoesnotexist/data/japan/wforms_GAN_input_v20220805.h5'
    outfile = 'thisquakedoesnotexist/data/japan/downsampled.h5'
    waveforms_file = 'thisquakedoesnotexist/data/japan/waveforms.npy'
    attribute_file = 'thisquakedoesnotexist/data/japan/attributes.csv'

    print('Loading ', input_file, ' ...')
    f = h5py.File(input_file,'r')

    pd.DataFrame(f)

    # initial time oaf recording
    time_ini_sec = 5.0

    # length of recording
    tt_wave_sec = 50.0
    sample_rate = 20.0

    # downssample factors
    factor = 5
    lower_bound = 4.0
    upper_bound = 7.5

    preprocessor = Preprocessor(input_file, outfile, time_ini_sec, tt_wave_sec, sample_rate)

    preprocessor.downsample(factor, lower_bound, upper_bound)
    preprocessor.compute_pga()
    preprocessor.save_to_files()


if __name__ == '__main__':
    main()

