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
        # else:
        #     self.directory = directory
        else:
            # self.open_raw(directory, verbose=verbose) # load the signal
            self.directory = directory # store for future info
        
        
        # probe information
        if isinstance(probe_map,(str,PathLike)):
            self.probe_map = kilosort.io.load_probe(probe_map)
        elif isinstance(probe_map,dict):
            self.probe_map = probe_map
        else:
            print(f'Could not load probe map. Will not be able to use that information for ERAASR or kilosort')


        # run ERAASR on raw data
        # self.ERAASR()
    
        

    def open_raw(self, directory:str = None, verbose:bool = False):
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
        if directory is None:
            directory = self.directory
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

        # continuous data -- this is going to be the biggest thing
        # iterate if we're potentially using a big hunk of memory:
        if not dir_too_big(directory):
            self.raw_sig = session.recordnodes[0].recordings[0].continuous[0].samples
        else:
            print("[open_raw] not enough space in memory, creating memmap")

            # preallocate the memmap array
            total_samples = recording.continuous[0].timestamps.shape[0]
            self.raw_sig = np.memmap(TemporaryFile("w+b"), dtype='float64',mode='w+',shape=[total_samples,64])

            # decide on number of loops -- let's go for 1/8 of available memory per loop
            svmem = virtual_memory()
            len_iter = svmem.available/(8*8*64) # 1/8 of available memory, 8 bytes per 64 bit float, 64 channels

            # pagination
            indices = np.append(np.arange(stop=total_samples, step=len_iter), total_samples)
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

        if 'raw_sig' not in dir(self):
            print('[ERAASR] raw signal has not been loaded. Loading now')
            self.open_raw()

        sos_filt = signal.butter(N = 2, Wn = [10], fs = self.metadata['fs'], btype='high', output='sos') # create filter

        # check memory usage and whether raw_sig is a memmap
        svmem = virtual_memory()
        not_enough_space = svmem.available/4 <= self.raw_sig.nbytes
        # if there isn't enough space or the raw_sig is a memmap, set up as memmaps
        if isinstance(self.raw_sig,np.memmap) or not_enough_space:
            print("[ERAASR] not enough space in memory, creating memmap for array")
            len_iter = int(svmem.available/(8*8*64)) # 1/8 of available memory, 8 bytes per 64 bit float, 64 channels

            # pagination
            total_samples = self.raw_sig.shape[0]
            indices = np.append(np.arange(stop=total_samples, step=len_iter), total_samples)
            indices = np.array((indices[:-1],indices[1:])).T

            # allocate array
            self.eraasr_cleaned = np.memmap(TemporaryFile("w+b"),
                                            dtype="float64",
                                            mode="w+",
                                            shape=self.raw_sig.shape)

        # otherwise set it up as a numpy array 
        else:
            indices = np.array([[0,self.raw_sig.shape[0]]])
            self.eraasr_cleaned = np.empty_like(self.raw_sig)


        # find close channels
        channel_mask = np.sqrt((self.probe_map['xc'][:,None] - self.probe_map['xc'])**2 
                               + (self.probe_map['yc'][:,None] - self.probe_map['yc'])**2) # 
        channel_mask = (channel_mask < 100) # only keep locations more than 100 um apart


        # clean it
        for idx in tqdm(indices, desc="[ERAASR] cleaning"):
            filt_sig = signal.sosfiltfilt(sos_filt, self.raw_sig[idx[0]:idx[1],:], axis=0) # filter it, place in new array

            # fit the principal components -- only looking for the first 4 (per 2018 paper)
            pca = PCA(n_components=4)
            pca.fit(filt_sig)

            # across-channel artifacts
            # sig_clean = np.empty_like(filt_sig) # make a copy for subtraction
            for channel in tqdm(np.arange(filt_sig.shape[1])):
                
                Wnc = pca.components_.copy() # make a copy for the exclusionary projection matrix
                Wnc[:,channel_mask[channel,:]] = 0 # exclude channel and neighbors'

                Ac = np.matmul(filt_sig, Wnc.T)
                # the normal equation is a lot faster than gradient descent!
                # ArtMat = np.linalg.lstsq(Ac,filt_sig[:,channel], rcond=None)[0]
                # self.eraasr_cleaned[idx[0]:idx[1],channel] = filt_sig[:,channel] - np.matmul(Ac,ArtMat)
                self.eraasr_cleaned[idx[0]:idx[1],channel] = filt_sig[:,channel] - np.matmul(Ac,np.matmul(np.pinv(Ac),filt_sig[:,channel]))


    def kilosort(self, version:int = 4, sig:str='eraasr_cleaned', filter:bool = True):
        '''
        kilosort
            runs kilosort on the data
        
        inputs:
            version : str                   kilosort version to use [4]
            sig : str                       'eraasr_cleaned' or 'raw_sig'
            filter : bool                   run a 250 - 6500 hz BPF over the signal [true]
        
        '''

        # check to make sure we got a valid signal name
        if sig not in ['eraasr_cleaned','raw_sig']:
            print('[kilosort] Only options for are eraasr_cleaned or raw_sig')
            return -1
        
        kilosort_settings = {'probe':self.probe_map,
                             'n_chan_bin': 64,
                             'filename':self.eraasr_filename}
        
        kilosort.run_kilosort(settings=kilosort_settings)
        



    def rot2vel(self, event_channels:dict = {'lead':3, 'follow':5, 'index':4}, vel_type:str = 'deg', interp:str = 'linear'):
        '''
        rot2vel
            takes the raw signal of the rotary encoder and turns it into an
            angular velocity. Default is deg/s.

        inputs:
            event_channels : list[int]      rotary encoder channels [1,2]
            vel_type : str                  degrees (deg) or radians (rad) per second ['deg']
            interp : str                    type of interpolation ['linear']
    
        outputs:
            ang_vel: np.array               Tx1 angular velocity 
        '''

        # if any([chan not in self.raw_events['channel'] for chan in event_channels]):
        pass       





# ------------------------------ 
    


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


def dir_too_big(directory:typing.Union[PathLike, str] = '.', memory_pct:float = 25) -> bool:
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