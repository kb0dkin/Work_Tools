#! /usr/bin/env python

# Open Ephys Utilities
# 
# These are some basic open/closing etc tasks 

from open_ephys.analysis import Session
import numpy as np
from scipy import signal
from os import path, listdir
from sklearn.decomposition import PCA
import pickle
import re
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from tqdm.notebook import tqdm


# ---------------------------------- #
def open_sig_stims(directory:str, verbose:bool = False, reconvert:bool = False, save:bool = True, save_name:str = 'raw_signal.pkl'):
    '''
    open_sig_stims:
        opens an open_ephys recording and returns a numpy array with the signal
        (TxN orientation) and a timestamp of the stimulation pulses

        will return data in only the first recordnode. Change as needed

    
    inputs:
        directory: str      - base directory of the recording
        verbose: bool       - how much information to share when loading the recording [False]
        reconvert: bool     - If a file with the saved signal already exists, do we reconvert? [False]
        save: bool          - Save the converted data? [True]

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

    # load a previously converted file
    if not reconvert and path.exists(path.join(directory, save_name)):
        print(f'loading previously converted file {path.join(directory,save_name)}')
        with open(path.join(directory, save_name), 'rb') as fid:
            data = pickle.load(fid)
            sig = data['sig']
            timestamps = data['timestamps']
            stim = data['stim']
            stim_ts = data['stim_ts']
        # skip the rest if it loaded properly
        if all([sig is not None, timestamps is not None, stim is not None, stim_ts is not None]):
            return sig, timestamps, stim, stim_ts
        else: # else fall back to conversions
            print(f'Could not load data from {save_name}. Loading from open-ephys files')

 

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
    stim_stream = recording.samples[:,64] - recording.samples[:,64].mean()
    stim = np.argwhere(np.diff(stim_stream>100) == 1) # find beginning and end of high values
    stim = stim[:-1] if stim.shape[0]%2 else stim # if we have a stimulus overlapping with the end, get rid of it
    stim_ts = recording.sample_numbers[stim] / recording.metadata['sample_rate'] # recording doesn't start at t=0
    stim = stim.reshape([int(stim.shape[0]/2),2]) # reshape to Ex2 (E == #stim)
    stim_ts = stim_ts.reshape([int(stim_ts.shape[0]/2),2]) # reshape to Ex2 (E == #stim)


    # timestamps -- 
    #   we'll be using "sample numbers" as the basis, which don't start at 0
    timestamps = recording.sample_numbers / recording.metadata['sample_rate']

    if save:
        print(f'saving data to {path.join(directory,save_name)}')
        with open(path.join(directory,save_name),'wb') as fid:
            data = {'sig':sig, 'timestamps':timestamps, 'stim':stim, 'stim_ts':stim_ts}
            pickle.dump(data,fid)


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
    ts_path = path.join(directory, 'timestamps.npy')
    stim_path = path.join(directory, 'stim.npy')
    stim_ts_path = path.join(directory, 'stim_ts.npy')

    raw_exists = path.exists(raw_path)
    ts_exists = path.exists(ts_path)
    stim_exists = path.exists(stim_path)
    stim_ts_exists = path.exists(stim_ts_path)

    if force == False and (raw_exists and ts_exists and  stim_exists and stim_ts_exists):
        print('Files have already been written. Skipping all')
        return -1

    sig, ts, stim, stim_ts = open_sig_stims(dir) # load the signals

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
def ERAASR(sig:np.array, chan_map:dict = None, num_surround:int = 0, fs:int = 30000, mode:str = 'ERAASR'):
    '''
    ERAASR
        implementing the PCA-based artifact rejection technique from O'Shea and Shenoy 2019
        
        Pre-filter data to get rid of obvious junk
        1. HPF at 10 hz (respiratory noise etc)

        Next, the across-channel removal
        2. PCA to get the matrix weighting
        3. Remove top 4 (adjusted PCs) from each channel c
            a. subtract Reconstructed PCs from array I guess. Seems a little indirect


    inputs:
        sig:np.array        - TxC array of the raw signal
        chan_map:dict       - channel map if accounting for surrounding channels in Wc [None]
        num_surround:int    - number of electrodes away from channel to remove from Wc [0]

    outputs:
        sig_clean:np.array  - TxC "cleaned" array. So far this is just looking at multi-channel, not multi-stimulus artifacts
    '''

    sos_filt = signal.butter(N = 2, Wn = [10], fs = fs, btype='high', output='sos') # create filter
    filt_sig = signal.sosfiltfilt(sos_filt, sig, axis=0) # filter it

    # fit the principal components -- only looking for the first 4 (per 2018 paper)
    pca = PCA(n_components=4)
    pca.fit(filt_sig)

    # loop through each channel
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

    return sig_clean


# ---------------------------------- #
def threshold_crossings(sig:np.array, fs:int = 30000, high_pass:int = 300, low_pass:int = 6000,
                        thresh_mult:float = -3, multi_rejection:int = 10, low_cutoff:int = -20):
    '''
    find threshold crossings based on a multiplier. First run a bandpass filter, 
    then do some basic artifact rejection.

        1. Bandpass filter [default 300-6000 hz]
        2. 

    
    inputs:
        sig : np.array          - input signal, TxN
        fs : int                - sample rate [30000]
        b_high : int            - HPF frequency in hz [300]
        b_low : int             - LPF frequency in hz [6000]
        thresh_mult : int       - standard deviation threshold multiplier [-3]
        multi_rejection : bool  - should we remove spikes that are on more than 10 channels in less than 1 ms?
    
    outputs:
        spikes : dict           - dataframe with spike times, channel #, and waveform

    '''

    # BPF
    sos_bpf = signal.butter(N = 8, Wn = [high_pass, low_pass], btype='bandpass', fs = fs, output='sos')
    sig_filt = signal.sosfiltfilt(sos=sos_bpf, x = sig, axis=0)

    # find the threshold values for each channel
    thresholds = np.std(sig_filt, axis=0) * thresh_mult
    # make the threshold at least ..
    if low_cutoff is not None: # if we want to make sure the waveforms are at least low_cutoff
        thresholds = np.where(thresholds<low_cutoff, thresholds, low_cutoff) # replace any threshold values that are less than the cutoff
    xings = np.nonzero(np.diff((sig_filt < thresholds).astype(int), axis=0) == 1)

    # need to introduce some basic cross-channel artifact rejection
    # if more than 10 channels have a spike in less than one ms assume it's not real
    if multi_rejection is not None:
        multi_chan = np.nonzero((xings[0][multi_rejection:] - xings[0][:-multi_rejection]) < (.001*fs))[0] # is the lapsed time less than fs?
        multi_chan = np.unique(multi_chan[:,np.newaxis] + np.arange(10)[np.newaxis,:])
        keep_chan = np.setdiff1d(np.arange(len(xings[0])), multi_chan)
        xings = (xings[0][keep_chan], xings[1][keep_chan])


    # split into per-channel dictionary
    bt = int(.0003*fs)
    at = int(.0012 * fs)
    spike_dict = {}
    for i_channel in np.arange(sig.shape[1]):
        spike_ts = xings[0][xings[1] == i_channel] # sample #
        # waveform -- make sure the whole thing is within the bounds of the signal
        spike_wf = [sig_filt[ts-bt:ts+at,i_channel] for ts in spike_ts if (ts>bt and ts+at<sig.shape[0])] # waveform
        spike_wf = np.array(spike_wf)

        spike_dict[i_channel] = {'sample_no':spike_ts, 'waveform':spike_wf}

    return spike_dict



# ---------------------------------- #
def calc_FR(spike_dict:dict, max_samp:int = None, min_samp:int = None, fs:int = 30000, bin_width:float = .01, count:bool = False):
    '''
    calculates the firing rates or spike counts (count == TRUE) for each channel. Not currently
    set to do any smoothing. May do that later

    inputs: [default]
        spike_dict:dict             - as output from threshold_crossings
        max_samp: int               - sample number for the max value of the binning array. Will look through the dictionary for max values if not provided [None]
        min_samp: int               - sample number for min value of the binning array.  [None]
        fs: int                     - sample frequency in hz [30000]
        bin_width : float           - bin width in seconds [.01]
        count : bool                - return spike counts (instead of firing rates) [False]

    outputs:
        firing_rates/spike_counts   - either the firing rates or counts per bin, depending on the count flag
    '''

    # calculate length of binning array
    if (max_samp is None) or (min_samp is None):
        print('Calculating bin range start and finish is a bummer! . Next time give me some info!')
        max_samp,min_samp = 0,0
        for data in spike_dict.values():
            max_samp = int(max(max_samp, data['sample_no'].max()))
            min_samp = int(min(min_samp, data['sample_no'].min()))

    bin_width = int(bin_width * fs) # bin width
    bins = np.arange(start = min_samp, step = bin_width, stop = max_samp + bin_width) # create an array of bins
    firing_rates = np.empty((len(bins)-1, len(spike_dict.keys()))) # pre-allocate array
    for channel, data in spike_dict.items():
        firing_rates[:,channel],_ = np.histogram(data['sample_no'], bins)

    if count: # currently we've actually got counts-per-bin, not firing rates
        return firing_rates, bins[:-1]
    else:
        return firing_rates / bin_width, bins[:-1] # divide by the bin width to give the firing rate



# ---------------------------------- #
def LFP_responses(sig, stims, len_ms:int = 25, n_chans:int = 64, sample_rate:int = 30000):

    samp_len = int(len_ms*(sample_rate/1000)) # ms * samp/ms

    # set up the stims to plot patches
    n_stims = stims.shape[0] # number of stimulation events

    responses = np.zeros((n_stims, samp_len, n_chans))
    mins = np.zeros((n_stims, n_chans))
    min_samp = np.zeros((n_stims, n_chans))

    for i_stim, stim in enumerate(stims):
        response = sig[stim[0]:stim[0]+samp_len,:]
        means = np.mean(sig[stim[0]+4:stim[1]-4,:], axis = 0)
        responses[i_stim,:,:] = response - means # response for each channel
    
        mins[i_stim,:] = np.min(response - means, axis=0)
        min_samp[i_stim,:] = np.argmin(response - means, axis=0)

    return mins, min_samp



# ---------------------------------- #
def LFP_stim_bulk(base_dir:str, n_chans:int = 64, save_fn:str = 'LFP_resp.pkl', save_plot:bool= True, reconvert:bool = False):
    '''
    Load the files, create a pandas dataframe of the responses, save and plot

    This setup will look for the current, distance, and pulsewidth in the directory names.

    inputs: [default]
        base_dir:str        - location of the base directory
        n_chans:int         - number of channels
        save_fn:str         - name to save the LFP response dataframe ['LFP_resp.pkl']. Does not save if None
        save_plot:bool      - should we save the output plots? [True]
        reconvert:bool      - should we reconvert recordings that have already been converted? [False]
    '''
    directories = [dd for dd in listdir(base_dir) if path.isdir(path.join(base_dir,dd)) and 'mA' in dd and 'skip' not in dd]

    # find unique current and distance values
    currents = np.unique(np.array([re.search('(\d{1,2})mA',dd)[1] for dd in directories]).astype(float))
    distances = np.unique(np.array([re.search('(\d?\.?\d)mm',dd)[1] for dd in directories]).astype(float))

    # set up a figure for mean responses for a couple of channels for each distance
    ax_map = dict(zip(distances, range(len(distances)))) # mapping the distance onto axis number
    rng = np.random.default_rng() # getting a random subset of the channels for plotting the average
    chan_examp = rng.choice(n_chans, 4)
    fig_means,ax_means = {},{}
    for chan in chan_examp: # setting up a dictionary of one figure per channel, one axis per distance
        fig_means[chan],ax_means[chan] = plt.subplots(ncols=len(distances))

    print(f'Found {len(directories)} recordings in {base_dir}')
    print(f'{len(currents)} unique current values and {len(distances)} unique distances')

    resp_df = pd.DataFrame(columns=['Channel_no','Current','Distance','uMin','uMin_ts'])
    for sub_dir in tqdm(directories):
        
        # get the current and distance value from the directory name
        current = float(re.search('(\d{1,2})mA', sub_dir)[1])
        distance = float(re.search('(\d?\.?\d)mm', sub_dir)[1])

        directory = path.join(base_dir, sub_dir) # go through the subdir, check to make sure it exists and that there's data inside

        # open the directory
        sig, timestamps, stims, stim_ts = open_sig_stims(directory, reconvert=reconvert)

        # skip it if we don't have any events
        print(f'{sub_dir}: {stims.shape}')

        # plot the mean response on the appropriate axis
        i_dist = ax_map[distance]
        for ch in chan_examp:
            plot_avg_LFP(sig, stims, channel=ch, ax=ax_means[ch][i_dist], label=f'{current} mA')

        # pull out the stim responses
        mins, min_samp = LFP_responses(sig, stims)

        # means and medians for each channel
        uMins = np.mean(mins, axis=0)
        uMins_ts = np.mean(min_samp, axis=0)/30000

        # a nested dictionary of all of the channels responses
        tdict = {ii:{'Channel_no':ii, 
                'Current':current,
                'Distance': distance,
                'uMin':uMins[ii],
                'uMin_ts':uMins_ts[ii],
                } for ii in range(n_chans)}

        t_df = pd.DataFrame.from_dict(tdict, orient='index') # create a dataframe

        resp_df = pd.concat([resp_df, t_df], ignore_index=True)

    # save it
    if save_fn is not None:
        with open(path.join(base_dir, save_fn), 'wb') as fid:
            pickle.dump(resp_df, fid)

    # plot everything
    fig_min = plt.figure()
    plot_LFP_mins(resp_df, fig_min)
    fig_time = plt.figure()
    plot_LFP_min_times(resp_df, fig_time)

    # clean up the average plots
    for fig_key, fig_value in  fig_means.items():
        fig_value.suptitle(f'Channel {ch} average responses, per distance and current')
        for ax_key, ax_value in ax_means[fig_key].keys():
            ax_value.legend()
            ax_value.set_title(f'Stimulation at {distances[ax_key]} mm')
    
    # should we save it?
    if save_plot is not None:
        fig_min.savefig(path.join(base_dir, 'minimum_value.svg'))
        fig_time.savefig(path.join(base_dir, 'minimum_time.svg'))

        for fig_key, fig_value in  fig_means.items():
            fig_value.savefig(path.join(base_dir, f'channel{fig_key}_mean_responses.svg'))



def plot_avg_LFP(sig, stims, len_ms:int = 25, channel = 0, ax:plt.axes=None, label:str=None):
    # Plot the average response for a particular channel
    
    if ax is None:
        fig,ax = plt.subplots()
    
    if label == None:
        label = f'Channel {channel}'

    # set up the stims to plot patches
    n_stims = stims.shape[0] # number of stimulations

    # put together a NxT array
    t_len = len_ms * 30
    responses = np.zeros((n_stims, t_len))
    
    # go through each stim
    for i_stim, stim in enumerate(stims):
        response = sig[stim[0]:stim[0]+len_ms*30,channel] # pop out the stimulation response
        art_means = np.nanmean(sig[stim[0]+4:stim[1]-4]) # center the during-stimulation to 0
        responses[i_stim,:] = response - art_means
    
    # put together the means, STDs, and timestamps
    ts = np.arange(t_len)/30
    means = np.nanmean(responses, axis=0)
    line = ax.plot(ts, means, label=label)
    
    # standard deviation of response
    ts_std = np.ravel(np.array([ts, ts[::-1]]))
    std = np.ravel(np.array([means + np.std(responses, axis=0), means[::-1] - np.std(responses, axis=0)[::-1]]))
    patch_array = np.array([ts_std, std]).T
    std_patch = Polygon(patch_array, alpha=0.2, color=line[-1].get_color())
    ax.add_patch(std_patch)

    # stimulation time
    stim_len = np.nanmean(np.diff(stims, axis=1))/30
    top,bottom = np.nanmax(responses), np.nanmin(responses)
    top,bottom = top + .2*abs(top), bottom - .2*abs(bottom)
    stim_patch = Polygon(np.array([[0, bottom],[stim_len, bottom], [stim_len, top], [0, top]]), alpha=0.2, color='k')
    ax.add_patch(stim_patch)




# ---------------------------------- #
def plot_LFP_mins(resp_df:pd.DataFrame, fig:plt.figure = None):
    '''
    plot the time for the minimum deviation as a function of the distance, per current value

    '''

    currents = resp_df.Current.unique()
    distances = resp_df.Distance.unique()
    currents.sort()
    distances.sort()
    
    if fig:
        fig,ax = plt.subplots(ncols=len(currents), sharex=True, sharey=True)

    for i_curr,curr in enumerate(currents):
        dist_cmp = resp_df.loc[resp_df.Current==curr ]
        ax[i_curr].scatter(dist_cmp.Distance, dist_cmp.uMin, s = 2, color='grey')
        current_means = dist_cmp.groupby('Distance').mean('uMin')
        ax[i_curr].plot(current_means.index, current_means.uMin, color='k')

        # ax[i_curr].legend([f'{current} mA' for current in currents], loc=4)
        ax[i_curr].set_title(f'LED current: {curr} mA')
        ax[i_curr].set_xlabel('Distance (um)')


        # remove the outer boxes
        for spine in ['top','bottom','right','left']:
            ax[i_curr].spines[spine].set_visible(False)
            # ax_time[i_chan].spines[spine].set_visible(False)
    

    ax[0].set_ylabel('Magnitude (uV)')
    fig.suptitle('Mean response minimum as a function of distance (per current level)')


# ---------------------------------- #
def plot_LFP_min_times(resp_df:pd.DataFrame, ax:np.array = None):
    '''
    plot the time for the minimum deviation as a function of current, per distance

    '''
    currents = resp_df.Current.unique()
    distances = resp_df.Distance.unique()
    currents.sort()
    distances.sort()

    if ax is None or ax.shape[1] != 4:
        fig,ax = plt.subplots(ncols=4)

    for i_dist,dist in enumerate(distances):
        curr_cmp = resp_df.loc[resp_df.Distance == dist]
        ax[i_dist].scatter(curr_cmp.Current, curr_cmp.uMin_ts, s = 2, color='gray')
        curr_means = curr_cmp.groupby('Current').mean('uMin_ts')
        ax[i_dist].plot(curr_means.index, curr_means.uMin_ts, color='k')

        ax[i_dist].set_title(f'Distance: {dist}um')
        ax[i_dist].set_xlabel('Current (mA)')


        # remove the outer boxes
        for spine in ['top','bottom','right','left']:
            ax[i_dist].spines[spine].set_visible(False)
    

    ax[0].set_ylabel('Time (ms)')
    fig.suptitle('Mean response minimum as a function of current (per distance)')




# ---------------------------------- #
def plot_peristim_FR(FR:np.array, bins:np.array, stims:np.array, channels:np.array = None, fs:int=30000, ax = None):
    '''
    Plot the mean and variance of FR of channels post stimulation
    
    '''
    
    # create an axis if not given 
    if ax == None:
        fig,ax = plt.subplots()

    # what channels are we working with?
    if channels is None:
        channels = np.arange(FR.shape[1])

    # split based on the stimulus time
    stim_length = int(min(np.diff(stims[:,0])))

    # go through each channel
    stim_response = 0
    for channel in channels:
        FR_mean = FR[:,channel].mean(axis=0)
        FR_std = FR[:,channel].std(axis=0)

    # get some timestamps






# ---------------------------------- #
def plot_PSTH(spike_dict:dict, stims:np.array, channel:int = 0, ax=None):
    '''
    Plot the PSTH for a single channel
    
    inputs:
        spike_dict:dict     - firing rates
        stims:np.array      - stim times
        channel:int         - which channel are we working with?
    '''
    # create an axis if not given
    if ax == None:
        fig,ax = plt.subplots()

    # split the spikes for the channel into a series of new channels
    spikes = spike_dict[channel]['sample_no']
    for i_stim in range(stims.shape[0]-1):
        spike_subset = spikes[np.logical_and(spikes>=stims[i_stim,0], spikes<stims[i_stim+1,0])] - stims[i_stim,0]
        ax.vlines(spike_subset,i_stim,i_stim+1)
        # create a patch at the stimulus point
        patch_array = np.array([[0, i_stim], [stims[i_stim,1]-stims[i_stim,0],i_stim],
                                [stims[i_stim,1]-stims[i_stim,0],i_stim+1], [0, i_stim+1]])
        stim_patch = Polygon(patch_array, alpha=0.4, color='k')
        ax.add_patch(stim_patch)



# ---------------------------------- #
def plot_spike_binary(spike_dict:dict, ax = None, stims:np.array = None):
    '''
    Plot an on/off of channels over time. show the stimulations if given
    '''

    # create an axis if not given
    if ax is None:
        fig,ax = plt.subplots()

    # for each channel, plot the spike times
    for channel, data in spike_dict.items():
        ax.vlines(data['sample_no'], channel, channel+1)
    
    # plot the stimulation times if given
    if stims is not None:
        for i_stim in range(stims.shape[0]):
            patch_array = np.array([[stims[i_stim,0],-1],
                                        [stims[i_stim,1],-1],
                                        [stims[i_stim,1],len(spike_dict.keys())],
                                        [stims[i_stim,0],len(spike_dict.keys())]])
            stim_patch = Polygon(patch_array, alpha=0.2, color='k')
            ax.add_patch(stim_patch)

    


# ---------------------------------- #
def plot_mean_waveforms(spike_dict:dict, channel:int=0, fs:int=30000, ax=None):
    '''
    plot the mean threshold crossing for a channel, and a patch around the standard deviation.

    could theoretically look at splitting into different units

    inputs:
        spike_dict
        channels
        std_flag
        map
    '''
    # create an axis if it doesn't exist
    if ax is None:
        fig,ax = plt.subplots()

    # go into the waveforms
    mean_wf = spike_dict[channel]['waveform'].mean(axis=0)
    std_= spike_dict[channel]['waveform'].std(axis=0)

    # create std patch
    std_array = np.ndarray((2*mean_wf.shape[0],2))
    std_array[:mean_wf.shape[0],:] = np.array(zip(mean_wf+std_, range(mean_wf.shape[0])))
    std_array[mean_wf.shape[0]::-1,:] = np.array(zip(mean_wf-std_, range(mean_wf.shape[0])))
    std_array[:,0] = std_array[:,0]/(fs/1000)
    std_patch = Polygon(std_array, alpha=0.2, color = 'o')

    # plot em
    ax.plot(mean_wf) 
    ax.add_patch(std_patch)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('magnitude (uV)')




# ---------------------------------- #
def plot_raw_filt():
    '''
    plot the raw and filtered data on a single plot
    
    '''
    pass


