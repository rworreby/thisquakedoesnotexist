import h5py
import numpy as np


class Downsampler:
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
    ):
        self.filename = filename
        self.outfile = outfile
        self.h5_file = h5py.File(self.filename, "r")
        self.data = h5py.File(self.outfile, "a")
        self.burnin = burnin
        self.duration = duration
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        return f"Downsampler instance for file: {self.filename}"

    def downsample(self, factor: int, threshold: float):
        """downsample_by_factor downsample data by a factor (frequency) and a threshold (magnitude).

        Selects all data with magnitude larger than the provided threshold,
        sampled at the frequency of :param factor:.

        :param factor: Factor by which the data is downsampled
        :type factor: int
        :param threshold: Threshold by which is downsampled
        :type threshold: float
        """

        magnitudes = self.h5_file["magnitude"][0]
        threshold_start = np.argmax(magnitudes > threshold)
        print(f"startpoint: {threshold_start}")

        for val in self.h5_file:
            ds_file = np.array(self.h5_file[val])
            dimension = len(self.h5_file[val].shape)

            begin = int(self.burnin * self.sample_rate)
            end = int(self.duration * self.sample_rate * factor + begin)
            if dimension == 1:
                self.data[val] = self.h5_file[val][begin:end:factor]
            elif dimension == 2:
                self.data[val] = self.h5_file[val][:, threshold_start:]
            elif dimension == 3:
                self.data[val] = self.h5_file[val][
                    :, begin:end:factor, threshold_start:
                ]
            else:
                print(f"Nothing to be done for {val}, skipping downsampling.")
                self.data[val] = self.h5_file[val][:, :]
                continue

            print(f"Downsampling for {val}:")
            print(f"Old dimensions of file: {self.h5_file[val].shape}")
            print(f"New dimension of file: {self.data[val].shape}")
