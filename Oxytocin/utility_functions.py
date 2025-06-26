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
from sklearn.decomposition import PCA
import re
import pandas as pd
from scipy.stats import ttest_ind
import kilosort
from glob import glob
import xmltodict
import typing

# file and OS operations
from os import path, listdir, PathLike, walk
from pathlib import Path
from psutil import virtual_memory
from tempfile import TemporaryFile

# plotting
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

# nice little status bars
from tqdm import tqdm



# ---------------------------------- #
class recording():
    '''
    Load the entire recording into a single class so that we don't have to pass data back and forth
    and create copies in memory. Hopefully this will reduce the memory usage.


    '''
    
    def __init__(self, directory: typing.Union[str, PathLike], probe_map:typing.Union[str,PathLike,dict] = None, verbose:bool = False):
        '''
        initialize the recording class. From the base directory of the recording we can
        pull in the data and all relevent settings.

        '''
        self.metadata = {'directory':directory,
                          'fs':30000,
                          }
        settings_file = glob(f'{directory}{path.sep}**{path.sep}settings.xml',recursive=True)

        # load the raw data
        if not path.exists(directory):
            print(f'{directory} does not exist. Choose a new directory then run the open_raw method')
        else:
            self.open_raw(directory, verbose=verbose) # load the signal
            self.directory = directory # store for future info
        
        
        # probe information
        if isinstance(probe_map,(str,PathLike)):
            self.probe_map = kilosort.io.load_probe(probe_map)
        elif isinstance(probe_map,dict):
            self.probe_map = probe_map
        else:
            print(f'Could not load probe map. Will not be able to use that information for ERAASR or kilosort')
    
        

    def open_raw(self, directory:str, verbose:bool = False):
        '''
        open_raw:
            opens an open_ephys recording and returns a numpy array with the signal
            (TxN orientation) and rotary encoder pulses

            will return data in only the first recordnode. Change as needed

    
        inputs:
            directory: str      - directory of the recording
            verbose: bool       - how much information to share when loading the recording [False]

        stores the data inside of the  "raw_sig", "raw_ts", and "raw_events" fields

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
    

        # load the continuous data from the first recordnode/recording
        recording = session.recordnodes[0].recordings[0]
        # timestamps
        self.raw_ts = recording.continuous[0].timestamps/30000 # hard coded because I'm not finding any info in the metadata. Maybe I need to open the settings file?
        # events -- this is important for both the stim times and the rotary encoder data
        self.raw_events = recording.events

        # # continuous data -- this is going to be the biggest thing
        # # iterate if we're potentially using a big hunk of memory:
        # if not dir_too_big(directory):
        #     self.raw_sig = session.recordnodes[0].recordings[0].continuous[0].samples
        # else:
        #     svmem = virtual_memory()
        #     print('Recording is too large for memory. Iterating to place it in memmap array')

        #     # preallocate the memmap array
        #     total_samples = recording.continuous[0].timestamps.shape[0]
        #     self.raw_sig = np.memmap(TemporaryFile("w+b"), dtype='float64',mode='w+',shape=[total_samples,64])

        #     # iterate in 10000 sample loops
        #     for i_idx,idx in tqdm(enumerate([[slc, slc+10000] for slc in np.arange(stop=total_samples, step=10000)])):
        #         self.raw_sig[idx,:] = recording.continuous[0].samples[idx,:] # pull things into memory in chunks
        #         self.raw_sig.flush() # write to disk


        svmem = virtual_memory()
        print('Recording is too large for memory. Iterating to place it in memmap array')

        # preallocate the memmap array
        total_samples = recording.continuous[0].timestamps.shape[0]
        self.raw_sig = np.memmap(TemporaryFile("w+b"), dtype='float64',mode='w+',shape=(total_samples,64))

        # iterate in 10000 sample loops
        indices = np.append(np.arange(stop=total_samples, step=100000), total_samples)
        indices = np.array((indices[:-1],indices[1:])).T
        for idx in tqdm(indices):
            self.raw_sig[idx[0]:idx[1],:] = recording.continuous[0].samples[idx[0]:idx[1],:] # pull things into memory in chunks
            self.raw_sig.flush() # write to disk


    

    def ERAASR(self):
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
        '''

        sos_filt = signal.butter(N = 2, Wn = [10], fs = self.fs, btype='high', output='sos') # create filter
        filt_sig = signal.sosfiltfilt(sos_filt, self.raw_sig, axis=0) # filter it

        # fit the principal components -- only looking for the first 4 (per 2018 paper)
        pca = PCA(n_components=4)
        pca.fit(filt_sig)

        # across-channel artifacts
        sig_clean = np.empty_like(filt_sig) # make a copy for subtraction
        for channel in tqdm(np.arange(filt_sig.shape[1])):
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
                print(f'Recording {i_rec} has:')
                print(f'\t{len(recording.continuous)} continuous streams')
                print(f'\t{len(recording.spikes)} spike streams')
                print(f'\t{len(recording.events)} event streams')
    
            print('\n')

    # load the continuous data from the first recordnode/recording
    recording = session.recordnodes[0].recordings[0].continuous[0]
    sig = recording.samples
    # sig = recording.samples[:,:64] * recording.metadata['bit_volts'][0] # convert to voltage


    # timestamps -- 
    #   we'll be using "sample numbers" as the basis, which don't start at 0
    # timestamps = recording.sample_numbers / recording.metadata['sample_rate']
    timestamps = recording.timestamps/30000 # hard coded because I'm not finding any info in the metadata. Maybe I need to open the settings file?

    # return it all
    return sig, timestamps


# ---------------------------------- #
def raw2nwb(directory:str, verbose:bool = False):
    '''
    raw2nwb
        takes raw recordings and turns it into an nwb file. This should allow us to convert
        once, then work on with the data on computers besides the 




    '''


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

    
    
def disk_usage(directory):
    '''
    utility function to see how much space all of the files in a directory are using.

    A quick and dirty way to decide whether we will use numpy memmapping and pagination for a recording
    '''
    total_size = 0
    for dirpath, dirnames, filenames in walk(directory):
        for f in filenames:
            fp = path.join(dirpath, f)
            # skip if it is symbolic link
            if not path.islink(fp):
                total_size += path.getsize(fp)

    return total_size





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


# ----------------------------------
# Utility functions
# ----------------------------------
def get_size(directory:typing.Union[PathLike, str] = '.'):
    # how much space to the recordings use?
    total_size = 0
    for dirpath, dirnames, filenames in walk(directory):
        for f in filenames:
            fp = path.join(dirpath, f)
            # skip if it is symbolic link
            if not path.islink(fp):
                total_size += path.getsize(fp)

    return total_size


def dir_too_big(directory:typing.Union[PathLike, str] = '.', memory_pct:float = 50) -> bool:
    # does the recording take more than X% of free memory?
    svmem = virtual_memory() # get memory usage information

    if memory_pct > 100:
        memory_pct = 50
        print(r"You provided a percentage greater than 100%. Working off 50% instead")

    return (svmem.available * (memory_pct/100)) <= get_size(directory) 





# ----------------------------------
# Run it as a script
# ----------------------------------
if __name__ == "__main__":
    '''
    
    '''