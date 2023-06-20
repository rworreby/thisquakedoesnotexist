#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def rolling_window(x, window_len):
    pos = 0
    while pos < len(x) - window_len:
        yield x[pos : pos+window_len]
        pos += 1


def plot_syn_data_single(samples, dist, dt, lt, fig_dir):
    tt = dt * np.arange(lt)

    for j in range(5):
        log_signal = np.log(np.abs(np.array(samples[j])))
        signal = np.array(samples[j])

        fig, ax = plt.subplots(4, 1, figsize=(15, 15))

        ax[0].plot(tt, signal, lw=0.5, label='Signal')
        # ax[0].plot(tt, signal_clean, color='r', lw=1, label='Clean Signal')
        ax[0].set_ylim([-1, 1])
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Normalized Amplitude')
        ax[0].legend()

        window_length = 40
        half_wl = int(window_length / 2)
        mean_data = np.convolve(log_signal, np.ones(window_length), 'valid') / window_length
        
        rw = rolling_window(log_signal, window_length)

        rolling_median = [None] * half_wl
        for i, window in enumerate(rw):
            rolling_median.append(np.median(window))

        ax[1].plot(tt, log_signal, lw=0.5, label='Log Transformed Signal')
        ax[1].plot(tt[half_wl:-half_wl+1], mean_data, label='Centered Rolling Mean')
        ax[1].plot(tt[:-half_wl], rolling_median, label='Centered Rolling Median')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Log Amplitude')
        ax[1].legend()

        rolling_max = [None] * half_wl
        rw = rolling_window(log_signal, window_length)

        for i, window in enumerate(rw):
            rolling_max.append(window.max())

        ax[2].plot(tt, log_signal, lw=0.5, label='Log Transformed Signal')
        ax[2].plot(tt[half_wl:], rolling_max, label='Centered Rolling Max')
        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylabel('Log Amplitude')
        ax[2].legend()

        s_fft = np.fft.rfft(signal)
        s_ps = np.real(s_fft * np.conj(s_fft))
        freq = np.fft.rfftfreq(signal.size, d=dt)

        rw = rolling_window(s_ps, window_length)
        rolling_mean = [None] * half_wl

        for i, window in enumerate(rw):
            rolling_mean.append(window.mean())

        ax[3].loglog(freq, s_ps, lw=0.5, label='Fourier Transformed Signal')
        ax[3].plot(freq[:-half_wl], rolling_mean, label='Centered Rolling Mean')
        ax[3].set_xlabel('Frequency [Hz]')
        ax[3].set_ylabel('Fourier Amplitude')
        ax[3].set_xlim((0.1, 50))
        ax[3].legend()

        fig.suptitle(f'Random Synthetic Waveform. Dist: {dist}')
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout()

        fig_file = os.path.join(fig_dir, f'syntetic_data_plot_{j}.png')
        plt.savefig(fig_file, format='png')


def plot_syn_data_grid(samples, distance, dt, fig_dir):
    n_rows = 12
    n_cols = 6
    n_tot = n_rows * n_cols 

    n_t = np.shape(samples[0])[0]
    tt = dt * np.arange(0, n_t)

    plt.figure()
    fig, axs = plt.subplots(n_rows, n_cols, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.2, 'wspace': 0.05},
                            figsize=(20,12),
                            )
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    for i, ax in enumerate(axs.flat):
        ax.set_ylim((-1, 1))
        ax.plot(tt, samples[i], linewidth=0.3)

#    plt.tight_layout()

    fig.suptitle(f'Randomly drawn samples from Generator. Distance: {distance} km')
    fig_file = os.path.join(fig_dir, 'syntetic_data_grid_plot.png')
    plt.savefig(fig_file, format='png')


def plot_waves_1C(ws, dt, t_max=50.0, ylim=None, fig_file=None, Np=12, fig_size = (6,9), stitle=None, show_fig=False):
    """
        Make plot of waves
        shape of the data has the form -> (Nobs, Nt)
        :param ws_p: numpy.ndarray of shape (Nobs, Nt)
        :param Np: number of 3C waves to show
        :param color: Color of the plot
        :param t_max: max time for the plot in sec
        :return: show a panel with the waves

        """
    Nt = int(t_max / dt)
    if ws.shape[0] < Np:
        # in case there are less observations than points to plot
        Np = ws.shape[0]

    ws_p = ws[:Np, :]
    
    # select waves to plot
    tt = dt * np.arange(Nt)
    plt.figure()
    fig, axs = plt.subplots(Np, 1, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.3, 'wspace': 0},
                            figsize=fig_size)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    for ik in range(Np):
        wf = ws_p[ik,:]
        axs[ik].plot(tt, wf, lw=0.5)
        if ylim is not None:
            axs[ik].set_ylim(ylim)

    for ax in axs.flat:
        ax.label_outer()

    if stitle is not None:
        fig.suptitle(stitle, fontsize=12)

    if show_fig:
        fig.show()

    if fig_file is not None:
        fformat = fig_file.split('.')[-1]
        print('saving:', fformat)
        fig.savefig(fig_file, format=fformat)
        if not show_fig:
            plt.close()
    else:
        fig.show()
    plt.close()
