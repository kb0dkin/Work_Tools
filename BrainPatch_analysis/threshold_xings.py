#! /usr/bin/env python

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_sig

from .openephys_utils import *



# Threshold crossings from a file
# 
# filter, pull out the spikes...


def threshold_extract(sig, fs:float = 30000, b_high:float = 300, b_low:float = 6000, thresh:float = -4.5, 
                      isi_min:float = .75, simul_max:int = 10, channels:int|list = -1):
    '''
    threshold_extract
        1. filters
        2. pulls out thresholds
        3. puts the threshold data into a separate numpy array

    inputs:
        sig : np.array      - input signal, TxN
        fs : float          - sample rate [30000]
        b_high : float      - HPF frequency in hz [300]
        b_low : float       - LPF frequency in hz [6000]
        thresh : float      - standard deviation threshold multiplier [-3]
        isi_min : float     - ISI time (in ms) to throw out spikes [.75]
        simul_max : int     - max simultaneous channels within 1ms window (throwout crosstalk) [10]
        channels : int|list - channels to record from; -1 == all [-1]
       
    outputs:
        spike_ts : np.array - Sx2 (S = spike count) timestamp and channel number for spike
        spike_wf : np.array - 1.5 ms waveform for each spike (300 us before, 1200 after)

    '''

    # pull out subset of channels if desired
    if channels != -1 and all([chan<sig.shape[1] for chan in channels]):
        sig = sig[:,channels]
    
    # filter
    filt_sos = signal.butter(N = 4, Wn = [b_high, b_low], btype = 'bandpass', fs = fs, output='sos')
    sig_filt = signal.sosfiltfilt(sos = filt_sos, x = sig, axis = 0)

    # thresholding
    thresholds = np.std(sig_filt, axis=0) * thresh # find the threshold value for each channel
    xings = np.where(np.diff(int(sig_filt < thresholds)) > 1)

    # remove unlikely units 
    #   too many simultaneous channels in less than a specific amount of time
    simul_starts = np.where(np.diff(xings[:,0], n=simul_max, axis=0) < (fs/1000))
    simul_mask = np.ones(xings.shape[0],1) # initialize a mask
    for ind in simul_starts: # remove all of the periods with too many multi-channel units
        simul_mask[simul_starts:simul_starts+simul_max] = 0
    xings = xings[simul_mask,:] # remove everything that happens too quickly

    # too short ISI on a single channel.
    valid_spikes = np.ndarray((0,2))
    waveforms = np.ndarray((0,int(1.5 * (fs/1000))))
    for chan in np.unique(xings[:,1]): # for each channel
        chan_spikes = xings[xings[:,1] == chan, :]
        chan_spikes = chan_spikes[[0, np.diff(chan_spikes) > isi_min * (fs/1000)+1],:]
        valid_spikes = valid_spikes.append(chan_spikes, axis=0)
        for spike in valid_spikes.iterrows():
            

    












