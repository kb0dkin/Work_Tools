#! /usr/bin/env python

# Open Ephys Utilities
# 
# These are some basic open/closing etc tasks 

from open_ephys.analysis import Session
import numpy as np
from scipy import signal
from os import path




# ---------------------------------- #
def open_sig_events(directory:str, verbose:bool = False):
    '''
    open_sig_events:
        opens an open_ephys recording and returns a numpy array with the signal
        (TxN orientation) and a timestamp of the stimulation pulses

        will return data in only the first recordnode. Change as needed

    
    inputs:
        directory: str      - base directory of the recording
        verbose: bool       - how much information to share when loading the recording

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
    recording = session.recordnodes[0].recordings[1].continuous[0]
    sig = recording.samples[:,:64] * recording.metadata['bit_volts'][0]

    # pull out stimulation events -- channel 64
    events = np.argwhere(np.diff(recording.samples[:,64]>5000) == 1) # find beginning and end of high values
    events_ts = recording.sample_numbers[events] / recording.metadata['sample_rate'] # recording doesn't start at t=0
    events = events.reshape([int(events.shape[0]/2),2]) # reshape to Ex2 (E == #events)
    events_ts = events_ts.reshape([int(events.shape[0]),2]) # reshape to Ex2 (E == #events)


    # timestamps -- 
    #   we'll be using "sample numbers" as the basis, which don't start at 0
    timestamps = recording.sample_numbers / recording.metadata['sample_rate']


    # return it all
    return sig, timestamps, events, events_ts



