#! /usr/bin/env python

# utility functions to open ephys recordings from an open_ephys
# rig, clean them up, and sort them using Kilosort 4. 
#
# This is currently going to be designed for Sara's setup
# for the oxytocin project, so the file organization structure
# will be following her lead.

from open_ephys.analysis import Session
import numpy as np
from scipy import signal
from os import path, listdir
from sklearn.decomposition import PCA
import re
import pandas as pd
from scipy.stats import ttest_ind
import kilosort

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from tqdm import tqdm



# ------------------------------ 
def open_raw(directory:str, verbose:bool = False):
    '''
    open_raw:
        opens an open_ephys recording and returns a numpy array with the signal
        (TxN orientation) and rotary encoder pulses

        will return data in only the first recordnode. Change as needed

    
    inputs:
        directory: str      - directory of the recording
        verbose: bool       - how much information to share when loading the recording [False]

    outputs:
        sig: np.array       - [TxN] (N==64) array, in uV
        ts: np.array        - [T] timestamps of signal
    '''

    # does the directory exist?
    if not path.exists(directory):
        print(f'{directory} does not exist')
        return -1

    # load in the data
    session = Session(directory)

    # give some information if asking for "verbose"
    if verbose:
        print(session)

        # iterate through the recording nodes
        for i_rec in range(len(session.recordnodes)):
            print(f'{len(session.recordnodes[i_rec].recordings)} recording(s) in session "{session.recordnodes[i_rec].directory}"\n')
            recordings = session.recordnodes[i_rec].recordings
    
            for i_rec,recording in enumerate(recordings):
                recording.load_continuous()
                recording.load_spikes()
                recording.load_events()
                recording.load_messages()

                print(f'Recording {i_rec} has:')
                print(f'\t{len(recording.continuous)} continuous streams')
                print(f'\t{len(recording.spikes)} spike streams')
                print(f'\t{len(recording.events)} event streams')
    
            print('\n')

    # load the continuous data from the first recordnode/recording
    recording = session.recordnodes[0].recordings[0].continuous[0]
    sig = recording.samples[:,:64] * recording.metadata['bit_volts'][0] # convert to voltage


    # timestamps -- 
    #   we'll be using "sample numbers" as the basis, which don't start at 0
    timestamps = recording.sample_numbers / recording.metadata['sample_rate']

    # return it all
    return sig, timestamps


# ---------------------------------- #
def rot2vel(rot_raw:np.array, input_fs:int = 30000, output_fs:int = 2000, vel_type:str = 'deg'):
    '''
    rot2vel
        takes the raw signal of the rotary encoder and turns it into an
        angular velocity. Default is deg/s.

    inputs [default]:
        rot_raw: np.array           raw values from the rotary encoder. Should be a Tx2+ array of high/low values
        input_fs: int               input sampling frequency in hz [30000 hz]
        output_fs : int             output sampling frequency in hz [2000 hz]
        vel_type : str              'deg' or 'rad' per second ['deg']
    
    
    outputs:
        ang_vel: np.array           Tx1 angular velocity 
    
    '''

    
    







# ---------------------------------- #
def ERAASR(sig:np.array, chan_map:dict = None, num_surround:int = 0, fs:int = 30000, mode:str = 'ERAASR', save:bool = True, save_dir:str = '.'):
    '''
    ERAASR
        implementing a modified version of the
        PCA-based artifact rejection technique from O'Shea and Shenoy 2019
        
        Pre-filter data to get rid of obvious junk
        1. HPF at 10 hz (respiratory movement etc)

        across-channel removal
        2. PCA to get the matrix weighting
        3. Remove top 4 (adjusted PCs) from each channel c
            a. subtract Reconstructed PCs from array
        
        across-stimulation removal (do we have anything similar for non-stim?)
        4. per-channel, PCA across stimulations
        5. reproject, subtract least-squares fit.  




    inputs:
        sig:np.array        - TxC array of the raw signal
        chan_map:dict       - channel map if accounting for surrounding channels in Wc [None]
        num_surround:int    - number of electrodes away from channel to remove from Wc [0]
        fs:int              - sample rate in Hz [30000]
        mode:str            - 'ERAASR' or 'mine' -- different methods of removing the projected artifact [True]
        save:bool           - save the file at {project_dir}\sig_eraasr.npy [True]
        save_dir:str        - where should we save it?

    outputs:
        sig_clean:np.array  - TxC "cleaned" array. So far this is just looking at multi-channel, not multi-stimulus artifacts
    '''

    sos_filt = signal.butter(N = 2, Wn = [10], fs = fs, btype='high', output='sos') # create filter
    filt_sig = signal.sosfiltfilt(sos_filt, sig, axis=0) # filter it

    # fit the principal components -- only looking for the first 4 (per 2018 paper)
    pca = PCA(n_components=4)
    pca.fit(filt_sig)

    # across-channel artifacts
    sig_clean = np.empty_like(filt_sig) # make a copy for subtraction
    for channel in np.arange(sig.shape[1]):
        Wnc = pca.components_.copy() # make a copy for the exclusionary projection matrix
        Wcc = Wnc[:,channel].copy() # and the channel-specific reprojection
        Wnc[:,channel] = 0 # exclude channel's contribution

        if mode == 'ERAASR':
            Ac = np.matmul(filt_sig, Wnc.T)
            ArtMat = np.linalg.lstsq(Ac,filt_sig[:,channel], rcond=None)[0]
            sig_clean[:,channel] = filt_sig[:,channel] - np.matmul(Ac,ArtMat)
        else:
            sig_clean[:,channel] = filt_sig[:,channel] - np.matmul(np.matmul(filt_sig, Wnc.T), Wcc)


    # # pull out across-stim artifacts
    # if stims is not None:
    #     # signal between stimuli, working with the minimum length.
    #     resp_len = np.diff(stims[:,0]).min()
    #     resp_ind = np.concatenate([np.arange(start=stim, stop=stim+resp_len) for stim in stims[:,0]])
    #     resp_sig = sig_clean[resp_ind,:].reshape((stims.shape[0], resp_len, sig_clean.shape[1])).transpose((1,0,2))

    #     # per-channel
    #     for channel_no in range(sig_clean.shape[1]):
    #         resp_sig = sig_clean[resp_ind,channel_no].reshape((stims.shape[0],  resp_len))
    #         # fit PCA (first two components) for the per-channel stimulation
    #         pca_stim = PCA(n_components = 2)
    #         pca_stim.fit(resp_sig)

    if save:
        np.save(path.join(save_dir, 'sig_eraasr.npy'), sig_clean)

    return sig_clean
