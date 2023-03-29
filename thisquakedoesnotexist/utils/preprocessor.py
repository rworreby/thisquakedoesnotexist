#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd


class Preprocessor:
    """ Simple class to downsample a fileset from a HDF5 file.

    If the dimension is 1, downsampling is done on the first dimension.
    If the dimension is 3, downsampling is done on the second dimension.
    Otherwise, no downsampling is performed.

    :param filename: Datafile name to read from
    :type filename: str
    :param outfile: Datafile name where the downsampled data is written to
    :type outfile: str
    :param burnin: Length of file at the beginning to be discarded (burn-in)
    :type burnin: float
    :param duration: Total length of file in seconds
    :type duration: float
    :param sample_rate: Number of samples per second
    :type sample_rate: float
    """

    def __init__(
        self,
        filename: str,
        outfile: str,
        burnin: float,
        duration: float,
        sample_rate: float,
        waveforms_file: str = 'thisquakedoesnotexist/data/japan/waveforms.npy',
        attributes_file: str = 'thisquakedoesnotexist/data/japan/attributes.csv',
    ):
        self.filename = filename
        self.outfile = outfile
        self.h5_file = h5py.File(self.filename, "r")
        self.data = h5py.File(self.outfile, "a")
        self.burnin = burnin
        self.duration = duration
        self.sample_rate = sample_rate
        self.waveforms_file = waveforms_file
        self.attributes_file = attributes_file

    def __repr__(self) -> str:
        return f"Downsampler instance for file: {self.filename}"

    def downsample(self, factor: int, lower_bound: float, upper_bound: float):
        """downsample_by_factor downsample data by a factor (frequency) and a threshold (magnitude).

        Selects all data with magnitudes between the provided lower and upper bounds,
        sampled at the frequency of factor.

        :param factor: Factor by which the data is downsampled
        :type factor: int
        :param lower_bound: Lower threshold from above which data is drawn
        :type threshold: float
        :param upper_bound: Upper threshold from below which data is drawn
        :type threshold: float
        """

        magnitudes = self.h5_file["magnitude"][0]
        threshold_start = np.argmax(magnitudes > lower_bound)
        threshold_end = np.argmax(magnitudes > upper_bound)
        print(f"Startpoint: {threshold_start}\nEndpoint: {threshold_end}")

        for val in self.h5_file:
            ds_file = np.array(self.h5_file[val])
            dimension = len(self.h5_file[val].shape)

            begin = int(self.burnin * self.sample_rate * factor)
            end = int(self.duration * self.sample_rate * factor + begin)
            if dimension == 1:
                self.data[val] = self.h5_file[val][begin:end:factor]
            elif dimension == 2:
                self.data[val] = self.h5_file[val][:, threshold_start:threshold_end]
            elif dimension == 3:
                self.data[val] = \
                    np.array(self.h5_file[val][:, begin:end:factor, threshold_start:threshold_end]).transpose((2, 0, 1))
            else:
                print(f"Nothing to be done for {val}, skipping downsampling.")
                self.data[val] = self.h5_file[val][:, :]
                continue

            print(f"Downsampling for {val}:")
            print(f"Old dimensions of data: {self.h5_file[val].shape}")
            print(f"New dimension of data: {self.data[val].shape}")
    
    
    def compute_pga(self):
        wfs_p = np.abs(self.data['waveforms'])
        
        # Calculate the max amplitude for each trace
        wa = np.max(wfs_p, axis=2)
        pga_v = np.max(wa, axis=1)
        
        self.data['pga_v'] = pga_v.reshape(1, -1)
        print(self.data.keys())


    def save_to_files(self):
        n_obs = np.array(self.data['waveforms']).shape[0]
        
        print(self.data.keys())
        for key in self.data.keys():
            print(self.data[key].shape)
        
        df = pd.DataFrame({
            'i_wf': np.arange(n_obs), # index for waveform
            'dist': self.data['hypocentral_distance'][0, :],  
            'ev_dep': self.data['hypocentre_depth'][0, :],
            'hypocentre_latitude': self.data['hypocentre_latitude'][0, :],
            'hypocentre_longitude': self.data['hypocentre_longitude'][0, :],
            'is_shallow_crustal': self.data['is_shallow_crustal'][0, :],
            'log10snr': self.data['log10snr'][0, :],
            'mag': self.data['magnitude'][0, :],
            'vs30': self.data['vs30'][0, :],
            'pga_v': self.data['pga_v'][0, :],
        })

        df.to_csv(self.attributes_file, index=False)

        np.save(self.waveforms_file, self.data['waveforms'])

        