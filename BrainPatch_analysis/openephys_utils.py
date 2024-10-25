#! /usr/bin/env python

# Open Ephys Utilities
# 
# These are some basic open/closing etc tasks 

from open_ephys.analysis import Session
import numpy as np
from scipy import signal
from os import path
from sklearn.decomposition import PCA




# ---------------------------------- #
def open_sig_events(directory:str, verbose:bool = False):
    '''
    open_sig_events:
        opens an open_ephys recording and returns a numpy array with the signal
        (TxN orientation) and a timestamp of the stimulation pulses

        will return data in only the first recordnode. Change as needed

    
    inputs:
        directory: str      - base directory of the recording
        verbose: bool       - how much information to share when loading the recording [False]

    outputs:
        sig: np.array       - [TxN] (N==64) array, in uV
        ts: np.array        - [T] timestamps of signal
        stim: np.array      - [Ex2] start and stop relative sample numbers for each stimulation pulse
        stim_ts: np.array   - [Ex2] start and stop timestamps for each stimulation pulse

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
    sig = recording.samples[:,:64] * recording.metadata['bit_volts'][0]

    # pull out stimulation stim -- channel 64
    stim = np.argwhere(np.diff(recording.samples[:,64]>5000) == 1) # find beginning and end of high values
    stim_ts = recording.sample_numbers[stim] / recording.metadata['sample_rate'] # recording doesn't start at t=0
    stim = stim.reshape([int(stim.shape[0]/2),2]) # reshape to Ex2 (E == #stim)
    stim_ts = stim_ts.reshape([int(stim_ts.shape[0]/2),2]) # reshape to Ex2 (E == #stim)


    # timestamps -- 
    #   we'll be using "sample numbers" as the basis, which don't start at 0
    timestamps = recording.sample_numbers / recording.metadata['sample_rate']


    # return it all
    return sig, timestamps, stim, stim_ts



# ---------------------------------- #
def save_raw_signal(directory:str, force:bool = False):
    '''
    save_filt_signal
        filter the signals and save into a numpy array as a couple of files.
        Overwrites any similarly named files if force == true

    inputs:
        directory: str      - base directory of open_ephys recording
        force: bool         - force an overwrite? [False]


    saved files:
        raw_sig.npy         - the raw signal (assumes 64 channels)
        timestamps.npy      - timestamps of each sample
        stim.npy            - stim windows in sample #s
        stim_ts.npy         - stim windows in seconds
    
    '''
    # file names, and check if they exist
    raw_path = path.join(directory, 'sig.npy')
    ts_path = path.join(directory, 'timestamps.py')
    stim_path = path.join(directory, 'stim.npy')
    stim_ts_path = path.join(directory, 'stim_ts.npy')

    raw_exists = path.exists(raw_path)
    ts_exists = path.exists(ts_path)
    stim_exists = path.exists(stim_path)
    stim_ts_exists = path.exists(stim_ts_path)

    if force == False and (raw_exists and ts_exists and  stim_exists and stim_ts_exists):
        print('Files have already been written. Skipping all')
        return -1

    sig, ts, stim, stim_ts = open_sig_events(dir) # load the signals

    # save it all
    if force == False and not raw_exists:
        np.save(raw_path, sig) # save the raw stuff
    else:
        print(f'{raw_path} already exists, skipping')

    if force == False and not ts_exists:
        np.save(ts_path, ts)
    else:
        print(f'{raw_path} already exists, skipping')

    if force == False and not stim_exists:
        np.save(stim_path, stim)
    else:
        print(f'{raw_path} already exists, skipping')

    if force == False and not stim_ts_exists:
        np.save(stim_ts_path, stim_ts)
    else:
        print(f'{raw_path} already exists, skipping')


# ---------------------------------- #
def save_filt_signal(directory:str, hpf:int = 300, lpf:int = 6000, fs:int = 30000, ntap:int = 8):
    '''
    save_filt_signal
        filter the signals and save into a numpy array as a couple of files

    output files:
        filt_sig.npy        - the filtered signal
        timestamps.npy      - timestamps (if doesn't already exist)
    
    '''
    if not path.exists(path.join(directory, 'raw_sig.npy')):
        sig, ts, stim, stim_ts = open_sig_events(directory) # load the signals
    else:
        np.load(path.join(directory, 'raw_sig.npy'))

    filt_sos = signal.butter(N=ntap, Wn=[hpf, lpf], btype='bandpass', fs=fs, output='sos') # build a filter
    filt_sig = signal.sosfiltfilt(filt_sos, sig, axis = 0) # run it through a bidirectional filter

    # save it all
    np.save(file=path.join(directory, 'filt_sig.npy'), arr = filt_sig) # save the filtered stuff
    if not path.exists(path.join(directory, 'timestamps.npy')): # save the timestamp file if it doesn't exist
        np.save(file=path.join(directory, 'timestamps.npy'), arr=ts)




# ---------------------------------- #
def plot_raw_filt():
    '''
    plot the raw and filtered data on a single plot
    
    '''
    pass

