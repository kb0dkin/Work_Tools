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
        filters, pulls out thresholds, and puts the threshold data into a separate numpy array

    inputs:
        sig : np.array      - input signal, TxN
        fs : float          - sample rate [30000]
        b_high : float      - HPF frequency in hz [300]
        b_low : float       - LPF frequency in hz [6000]
        thresh : float      - standard deviation threshold multiplier [-3]
        isi_min : float     - ISI time (in usec) to throw out spikes [.75]
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












