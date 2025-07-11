#! /usr/bin/env python

# Open Ephys Utilities
# 
# These are some basic open/closing etc tasks 

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

from tqdm.notebook import tqdm




# ---------------------------------- #
def open_sig_stims(directory:str, verbose:bool = False, reconvert:bool = False, save:bool = True):
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
    filenames = ['sig_raw.npy', 'stim_sample_nums.npy', 'stim_timestamps.npy', 'timestamps.npy'] # list of saved filenames if this has been converted
    if not reconvert and all([path.exists(path.join(directory, fn)) for fn in filenames]):
        if verbose:
            print(f"loading previously converted files {path.join(directory,'sig_raw.npy')}")

        sig = np.load(path.join(directory, 'sig_raw.npy'))
        stim = np.load(path.join(directory, 'stim_sample_nums.npy'))
        stim_ts = np.load(path.join(directory, 'stim_timestamps.npy'))
        timestamps = np.load(path.join(directory, 'timestamps.npy'))
        
        if all([sig is not None, timestamps is not None, stim is not None, stim_ts is not None]):
            return sig, timestamps, stim, stim_ts

    
    
    if verbose:
        print(f"Could not load data from {directory}. Loading from open-ephys files")

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

    # pull out stimulation events
    # do we have an event channel? If so, use that. Otherwise, it was recorded by the auxiliary channel
    if len(session.recordnodes[0].recordings[0].events) > 5: # arbitrary non-zero number:
        stim = session.recordnodes[0].recordings[0].events.sample_number.values - recording.sample_numbers[0] # remove sample# offset
    else:
        stim_stream = recording.samples[:,64] - recording.samples[:,64].mean()
        cutoff = stim_stream.max() * .9
        stim = np.argwhere(np.diff(stim_stream>cutoff) == 1) # find beginning and end of high values
    
    # put into a nice array, get the "timestamp" numbers too
    stim = stim[:-1] if stim.shape[0]%2 else stim # if we have a stimulus overlapping with the end, get rid of it
    stim_ts = recording.sample_numbers[stim] / recording.metadata['sample_rate'] # recording doesn't start at t=0
    stim = stim.reshape([int(stim.shape[0]/2),2]) # reshape to Ex2 (E == #stim)
    stim_ts = stim_ts.reshape([int(stim_ts.shape[0]/2),2]) # reshape to Ex2 (E == #stim)


    # timestamps -- 
    #   we'll be using "sample numbers" as the basis, which don't start at 0
    timestamps = recording.sample_numbers / recording.metadata['sample_rate']

    if save:
        if verbose:
            print(f'saving data to {directory}')
        np.save(path.join(directory, 'sig_raw.npy'), sig)
        np.save(path.join(directory, 'stim_sample_nums.npy'), stim)
        np.save(path.join(directory, 'stim_timestamps.npy'), stim_ts)
        np.save(path.join(directory, 'timestamps.npy'), timestamps)


    # return it all
    return sig, timestamps, stim, stim_ts



# ---------------------------------- #
def ERAASR(sig:np.array, stims:np.array = None, chan_map:dict = None, num_surround:int = 0, fs:int = 30000, mode:str = 'ERAASR', save:bool = True, save_dir:str = '.'):
    '''
    ERAASR
        implementing a modified version of the
        PCA-based artifact rejection technique from O'Shea and Shenoy 2019
        
        Pre-filter data to get rid of obvious junk
        1. HPF at 10 hz (respiratory noise etc)

        across-channel removal
        2. PCA to get the matrix weighting
        3. Remove top 4 (adjusted PCs) from each channel c
            a. subtract Reconstructed PCs from array
        
        across-stimulation removal (if stimulation time is available)
        4. per-channel, PCA across stimulations
        5. reproject, subtract least-squares fit.  




    inputs:
        sig:np.array        - TxC array of the raw signal
        stims:np.array      - Tx2 array of stimulation start and stop times [None]
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


# ---------------------------------- #
def kilosort_FR(kilosort_dir:str, bin_sec:float = .01, fs:int = 30000):
    '''
    Return the firing rates for each template

    inputs:
        kilosort_dir:str        - directory where the kilosort results were stored
        bin_sec:float           - bin length in seconds [.01]
        fs:int                  - sampling rate [30000]

    outputs:
        firing_rates:np.array   - TxN array of firing rates of N templates
    '''
    spike_times = np.load(path.join(kilosort_dir, 'spike_times.npy'))
    spike_templates = np.load(path.join(kilosort_dir, 'spike_templates.npy'))

    # bin times defined by sample #s, not seconds
    bin_times = np.arange(np.ceil(spike_times[-1]), step=bin_sec*fs)
    # empty array to fill later
    firing_rates = np.ndarray((bin_times.shape[0]-1, np.unique(spike_templates).shape[0]))
    for template in np.unique(spike_templates): # for each template...
        firing_rates[:,template] = np.histogram(spike_times[spike_templates == template], bins=bin_times)[0]

    return firing_rates/bin_sec, bin_times[:-1]/fs


# ---------------------------------- #
def gimme_a_FR_df(directories, probe_name, settings):
    # open the probe
    probe = kilosort.io.load_probe(probe_name)


    # sos for a BPF
    filtered = True # should we filter the data?
    sos = signal.butter(N = 8, Wn = [300, 6000], btype='bandpass', fs=30000, output='sos') if filtered else None

    # binning decisions
    bin_sec = 0.002

    # to create a pandas array later I guess
    FR_list = [] # this will contain the pre-stim mean, the mean stim response, and the map info

    # length of interest for the responses and the "low mean firing rate"
    resp_len = 10 # in bins
    low_fr_cutoff = 1


    # for use later
    color_inds_map = ['low_firing','inhibited','excited']

    for directory in tqdm(directories):

        # pull out the current and distance
        current = int(re.search('(\d{1,2})mA', directory)[1])
        distance = int(re.search('(\d{3,4})um', directory)[1])

        distances = np.sqrt(distance**2 + (probe['xc'] - probe['xc'].mean())**2)


        # filter at 300, 6000
        if filtered:
            filt_path = path.join(directory, 'sig_filter.npy')
            if not path.exists(filt_path):
                sig_filter = signal.sosfiltfilt(sos, sig_eraasr, axis=0)
                np.save(filt_path, sig_filter)


        # kilosort
        if not filtered:
            # looking at the eraasr location
            eraasr_path = path.join(directory, 'sig_eraasr.npy')
            results_dir= path.join(directory, 'kilosort4_unfiltered')
            
            if not path.exists(results_dir): # run it if it doesn't already exist. Otherwise just use the existing
                if not path.exists(eraasr_path):
                    sig, timestamps, stims, stim_ts = open_sig_stims(directory)
                    # clean, pull out threshold crossings, get firing rates
                    sig_eraasr = ERAASR(sig, save_dir=directory)
                else:
                    sig_eraasr = np.load(eraasr_path)
                
                settings['filename'] = eraasr_path
                kilosort.run_kilosort(settings, file_object=sig_eraasr.astype(np.float32), data_dtype='float32', results_dir=results_dir)

            else:
                print(f'{results_dir} already exists, using existing files')

        else:
            filt_path = path.join(directory, 'sig_filter.npy')
            results_dir=path.join(directory, 'kilosort4_filtered')

            if not path.exists(results_dir): # run it if it doesn't already exist. Otherwise just use the existing
                if not path.exists(filt_path):
                    sig, timestamps, stims, stim_ts = open_sig_stims(directory)
                    # clean, pull out threshold crossings, get firing rates
                    sig_eraasr = ERAASR(sig, save_dir=directory)
                    sig_filter = signal.sosfiltfilt(sos, sig_eraasr, axis=0)
                else:
                    sig_filter = np.load(filt_path)

                settings['filename'] = path.join(directory,'sig_filter.npy')
                kilosort.run_kilosort(settings, file_object=sig_filter.astype(np.float32), data_dtype='float32', results_dir=results_dir)

            else:
                print(f'{results_dir} already exists, using existing files')

        # firing rates from kilosort
        firing_rate, bins = kilosort_FR(results_dir, bin_sec=bin_sec)

        # get the channel-to-template map
        template_wf = np.load(path.join(results_dir, 'templates.npy'))
        channel_best = (template_wf**2).sum(axis=1).argmax(axis=-1) # find the channel with biggest variance

        # get stimulation times in bin count
        stims_bin = (stims/(30000*bin_sec)).astype(int)

        # pre-stimulation data
        prestim_means = firing_rate[:stims_bin[0,0], :].mean(axis=0) # put into firing rates instead of counts
        prestim_std = firing_rate[:stims_bin[0,0], :].std(axis=0) # put into firing rates instead of counts

        # get the means for 5 bins after the stims
        # indices for 5 ms after end of stimulation
        poststim_inds = np.array([np.arange(start=stim, stop=stim+resp_len) for stim in stims_bin[:,1]]).flatten()
        # split out that portion of the array and reshape it to a Stims x Time x Templates array
        poststim_resp = firing_rate[poststim_inds,:].reshape((stims.shape[0], resp_len, firing_rate.shape[1]))
        # find the mean
        poststim_mean = poststim_resp.mean(axis=0)

        # # split into three colors -- minimal fr before, fr increases, fr decreases
        color_inds = 1 + (poststim_mean.mean(axis=0)>prestim_means).astype(int)
        color_inds[prestim_means < low_fr_cutoff] = 0

        for template,channel in enumerate(channel_best):
            significant = ttest_ind(firing_rate[:stims_bin[0,0], template], poststim_resp[:,0,template], equal_var = False, alternative = 'less')
            
            
            FR_info = {'template':template,
                    'channel': channel,
                    'prestim_mean': prestim_means[template],
                    'prestim_std' : prestim_std[template],
                    'poststim_max': poststim_mean[:,template].max(),
                    'poststim_first':poststim_mean[0,template],
                    'type': color_inds_map[color_inds[template]],
                    'distance': distances[channel],
                    'depth': -probe['yc'][channel],
                    'current': current,
                    'shank_no':probe['kcoords'][channel],
                    'template_wf':template_wf[template,:,channel],
                    'mean_firing_rate':poststim_mean[:,template],
                    'recording':directory,
                    'significant':significant}
        
            FR_list.append(FR_info)


    FR_df = pd.DataFrame(FR_list)

    return FR_df

# ---------------------------------- #
def threshold_crossings(sig:np.array, fs:int = 30000, high_pass:int = 300, low_pass:int = 6000,
                        thresh_mult:float = -3, multi_rejection:int = None, low_cutoff:int = -20):
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
        spikes : dict           - dict with spike times, channel #, and waveform

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
def calc_FR(spike_dict:dict, max_samp:int = None, min_samp:int = None, fs:int = 30000, bin_sec:float = .01, bin_ms:float = None, count:bool = False):
    '''
    calculates the firing rates or spike counts (count == TRUE) for each channel. Not currently
    set to do any smoothing. May do that later

    inputs: [default]
        spike_dict:dict             - as output from threshold_crossings
        max_samp: int               - sample number for the max value of the binning array. Will look through the dictionary for max values if not provided [None]
        min_samp: int               - sample number for min value of the binning array.  [None]
        fs: int                     - sample frequency in hz [30000]
        bin_sec : float             - bin width in seconds [.01]
        bin_ms : float              - bin width in ms [None]
        count : bool                - return spike counts (instead of firing rates) [False]

    outputs:
        firing_rates/spike_counts   - either the firing rates or counts per bin, depending on the count flag
        bins                        - the beginning of each bin, in sample numbers (can convert to timestamps by dividing by fs)
    '''
    # overwrite bin_sec with bin_ms if given
    if bin_ms is not None:
        bin_sec = bin_ms / 1000

    # calculate length of binning array
    if (max_samp is None) or (min_samp is None):
        print('Calculating bin range start and finish is a bummer! . Next time give me some info!')
        max_samp,min_samp = 0,0
        for data in spike_dict.values():
            max_samp = int(max(max_samp, data['sample_no'].max()))
            min_samp = int(min(min_samp, data['sample_no'].min()))

    bin_width = int(bin_sec * fs) # bin width converted to samples
    bins = np.arange(start = min_samp, step = bin_width, stop = max_samp + bin_width) # create an array of bins
    firing_rates = np.empty((len(bins)-1, len(spike_dict.keys()))) # pre-allocate array
    for channel, data in spike_dict.items():
        firing_rates[:,channel],_ = np.histogram(data['sample_no'], bins)

    if count: # currently we've actually got counts-per-bin, not firing rates
        return firing_rates, bins[:-1]
    else:
        return firing_rates / bin_sec, bins[:-1] # divide by the bin width to give the firing rate



# ---------------------------------- #
def LFP_responses(sig, stims, len_ms:int = 25, n_chans:int = 64, sample_rate:int = 30000):

    samp_len = int(len_ms*(sample_rate/1000)) # ms * samp/ms

    # set up the stims to plot patches
    n_stims = stims.shape[0] # number of stimulation events

    responses = np.zeros((n_stims, samp_len, n_chans))
    mins = np.zeros((n_stims, n_chans))
    min_samp = np.zeros((n_stims, n_chans))

    for i_stim, stim in enumerate(stims):
        if stim[0]+4 >= stim[1]-4 or stim[1] > sig.shape[0]: # skip any where we might get an empty selection
            responses[i_stim,:] = 0
        else:
            response = sig[stim[0]:stim[0]+samp_len,:]
            means = np.mean(sig[stim[0]+4:stim[1]-4,:], axis = 0)
            responses[i_stim,:,:] = response - means # response for each channel
    
            mins[i_stim,:] = np.min(response - means, axis=0)
            min_samp[i_stim,:] = np.argmin(response - means, axis=0)

    return mins, min_samp



# ---------------------------------- #
def LFP_stim_bulk(base_dir:str, n_chans:int = 64,  save_plot:bool= True, reconvert:bool = False):
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
    currents = np.unique(np.array([re.search('(\d{1,2})mA',dd)[1] for dd in directories]).astype(int))
    distances = np.unique(np.array([re.search('(\d{3,4})um',dd)[1] for dd in directories]).astype(int))

    # set up a figure for mean responses for a couple of channels for each distance
    ax_map = dict(zip(distances, range(len(distances)))) # mapping the distance onto axis number
    patch_flag = {dist:currents.shape[0] for dist in distances} # only show one stimulation patch per mean
    rng = np.random.default_rng() # getting a random subset of the channels for plotting the average
    chan_examp = rng.choice(n_chans, 4)
    fig_means,ax_means = {},{}
    for chan in chan_examp: # setting up a dictionary of one figure per channel, one axis per distance
        fig_means[chan],ax_means[chan] = plt.subplots(ncols=len(distances), sharey = True)

    print(f'Found {len(directories)} recordings in {base_dir}')
    print(f'{len(currents)} unique current values and {len(distances)} unique distances')

    resp_list = []
    cur_dist_list = [] # so that we only do one recording per combination
    for sub_dir in tqdm(directories):
        
        # get the current and distance value from the directory name
        current = int(re.search('(\d{1,2})mA', sub_dir)[1])
        distance = int(re.search('(\d{3,4})um', sub_dir)[1])

        cur_dist = f'{current}_{distance}'

        directory = path.join(base_dir, sub_dir) # go through the subdir, check to make sure it exists and that there's data inside

        # open the directory
        sig, timestamps, stims, stim_ts = open_sig_stims(directory, reconvert=reconvert)

        # plot the mean response on the appropriate axis
        i_dist = ax_map[distance]
        if cur_dist not in cur_dist_list: # skip it if we already have this combo in the main plot
            for ch in chan_examp:
                # plot_mean_LFP(sig, stims, channel=ch, ax=ax_means[ch][i_dist], label=f'{current} mA', show_stim = patch_flag[distance] == 1)
                plot_mean_LFP(sig, stims, channel=ch, ax=ax_means[ch][i_dist], label=f'{current} mA', show_stim = 0)

        # patch_flag[distance] -= 1 # make it so the stimulation patch only shows up once

        # pull out the stim responses
        mins, min_samp = LFP_responses(sig, stims)

        # means and medians for each channel
        uMins = np.mean(mins, axis=0)
        uMins_ts = np.mean(min_samp, axis=0)/30000

        tlist = [{'Channel_no':ii, 
                'Current':current,
                'Distance': distance,
                'uMin':uMins[ii],
                'uMin_ts':uMins_ts[ii],
                } for ii in range(n_chans)]

        resp_list.extend(tlist)
        cur_dist_list.append(cur_dist)
    
    print(cur_dist_list)


    # turn the list into a dataframe -- much faster than concat
    resp_df = pd.DataFrame(resp_list) 


    # plot everything
    fig_min = plot_LFP_mins(resp_df)
    fig_min.set_size_inches((12,6))
    fig_time = plot_LFP_min_times(resp_df)
    fig_time.set_size_inches((12,6))

    # clean up the average plots
    for fig_key, fig_value in  fig_means.items():
        fig_value.suptitle(f'{fig_key} average responses, per distance and current')
        fig_value.set_size_inches((12,6))

        for ax_mean_i, ax_mean in enumerate(ax_means[fig_key]): # arange each axis properly
            # sort the legend order
            handles, labels = ax_mean.get_legend_handles_labels()
            order = np.argsort([int(re.search('(\d{1,2}) mA', label)[1]) for label in labels if 'mA' in label])
            ax_mean.legend([handles[o] for o in order], [labels[o] for o in order])
            # set titles, labels etc
            ax_mean.set_title(f'{distances[ax_mean_i]} mm')
            ax_mean.set_xlabel('Time (ms)')
            ax_mean.set_xlim([-2, 27])
            if ax_mean_i == 0:
                ax_mean.set_ylabel('Voltage (uV)')
            

        
    
    # should we save it?
    if save_plot is not None:
        fig_min.savefig(path.join(base_dir, 'minimum_value.svg'))
        fig_time.savefig(path.join(base_dir, 'minimum_time.svg'))

        for fig_key, fig_value in  fig_means.items():
            fig_value.savefig(path.join(base_dir, f'channel{fig_key}_mean_responses.svg'))



# ---------------------------------- #
def mean_stim_FR(firing_rate:np.array, stims:np.array, bin_sec:float = .001, bin_ms:float = None, normalize:bool = True, resp_length_samp:int = 12000, fs:int=30000):
    '''
    the average firing rate for a recording (per channel) for each recording setup
    
    inputs:
        firing_rate : np.array              - array of the firing rates
        stims : np.array                    - stimulation start and stop times (Nx2) 
        normalize : bool                    - Should we normalize so the average (pre-stim) firing rate is 1?
        fs : int                            - sample rate (Hz) [30000]
        bin_sec : float                     - bin window length (s) [.001]
        bin_ms : float                      - bin window length, overrules bin_sec if given (ms) [None] 
        resp_length_samp : int              - length of interest for response (samples) [15000]

        
    outputs:
        mean_fr:np.array            - 
        std_fr:np.array
    
    '''
    # overwrite bin_sec if we are given bin_ms
    if bin_ms is not None:
        bin_sec = bin_ms/1000

    # convert bin_win (seconds) into a sample value
    bin_samp = int(fs*bin_sec)

    # normalize off the average firing rates before the first stim:
    if normalize:
        pre_stim = firing_rate[:stims[0,0], :] # pull out the firing rates before the first stimulation
        firing_rate = np.matmul(firing_rate, np.linalg.pinv(pre_stim.mean(axis=0)*np.eye(firing_rate.shape[1]))) 

    # pull out the stimulation segments, then reshape
    rebase_stims = (stims/bin_samp).astype(int) # get the bin numbers, instead of sample numbers
    resp_length_bin = int(resp_length_samp/bin_samp) # get length of response in bins instead of samples
    stim_resp = firing_rate[np.concatenate([np.arange(start=row, stop=row+resp_length_bin).astype(int) for row in rebase_stims[:,0]]), :] 
    stim_resp = stim_resp.reshape((stims.shape[0],resp_length_bin,64)).transpose((1,0,2))

    
    # then pull out the mean for each channel and the std
    mean_fr = stim_resp.mean(axis=1) # mean across trials
    std_fr = stim_resp.std(axis=1) # standard deviation across trials

    return mean_fr, std_fr


# ---------------------------------- #
def plot_mean_LFP(sig, stims, len_ms:int = 25, pre_stim:int=0, channel = 0, ax:plt.axes=None, label:str=None, show_stim:bool = True, align_stim:bool=True):
    '''
    plot the average LFP value for a channel
    
    inputs : [default]
        sig : np.array          - filtered etc signal (samples x channels)
        stims : np.array        - stimulation start/stop sample numbers (stims x 2)
        len_ms : int            - window of interest (ms) [25]
        pre_stim : int          - pre-stimulation period to show (ms) [0]
        channel : int           - channel of interest [0]
        ax : plt.axes           - axis for plot [None]
        label : str             - label name for the LFP [None]
        show_stim : bool        - whether to show the stimulation or not [True]
        align_stim : bool       - subtract by the mean during the stimulus? [True]
        
    '''
    
    if ax is None:
        fig,ax = plt.subplots()
    
    if label == None:
        label = f'Channel {channel}'

    # set up the stims to plot patches
    n_stims = stims.shape[0] # number of stimulations

    # put together a NxT array
    t_len = (len_ms+pre_stim) * 30
    responses = np.zeros((n_stims, t_len))
    
    # go through each stim
    for i_stim, stim in enumerate(stims):
        if stim[0]+4 >= stim[1] or stim[1] > sig.shape[0]: # skip any where we might get an empty selection
            responses[i_stim,:] = np.nan
        else:
            response = sig[stim[0]-pre_stim*30:stim[0]+len_ms*30,channel] # pop out the stimulation response
            art_means = np.nanmean(sig[stim[0]+4:stim[1]-4]) if align_stim else np.nanmean(sig[stim[0]-pre_stim*30])
            responses[i_stim,:] = response - art_means
    
    # put together the means, STDs, and timestamps
    ts = np.arange(start=-pre_stim, stop=len_ms, step=1/30)
    means = np.nanmean(responses, axis=0)
    line = ax.plot(ts, means, label=label)
    
    # standard deviation of response
    ts_std = np.ravel(np.array([ts, ts[::-1]]))
    std = np.ravel(np.array([means + np.std(responses, axis=0), means[::-1] - np.std(responses, axis=0)[::-1]]))
    patch_array = np.array([ts_std, std]).T
    std_patch = Polygon(patch_array, alpha=0.2, color=line[-1].get_color())
    ax.add_patch(std_patch)

    # stimulation time
    if show_stim:
        stim_len, stim_count = np.unique(np.diff(stims, axis=1)/30, return_counts=True)
        stim_len = stim_len[np.argmax(stim_count)]
        # top,bottom = np.nanmax(responses), np.nanmin(responses)
        # top,bottom = top + .2*abs(top), bottom - .2*abs(bottom)
        bottom,top = ax.get_ylim()
        stim_patch = Polygon(np.array([[0, bottom],[stim_len, bottom], [stim_len, top], [0, top]]), alpha=0.2, color='k', label='stimulation period')
        ax.add_patch(stim_patch)




# ---------------------------------- #
def plot_LFP_mins(resp_df:pd.DataFrame):
    '''
    plot the time for the minimum deviation as a function of the distance, per current value

    inputs: [default]
        resp_df: pd.DataFrame           - 

    outputs: 
        figure
    '''

    currents = resp_df.Current.unique()
    distances = resp_df.Distance.unique()
    currents.sort()
    distances.sort()
    
    fig,ax = plt.subplots(ncols=len(currents), sharex=True, sharey=True)

    for i_curr,curr in enumerate(currents):
        dist_cmp = resp_df.loc[resp_df.Current==curr ]
        ax[i_curr].scatter(dist_cmp.Distance, dist_cmp.uMin, s = 2, color='grey')
        current_means = dist_cmp.groupby('Distance').mean('uMin')
        ax[i_curr].plot(current_means.index, current_means.uMin, color='k')

        # ax[i_curr].legend([f'{current} mA' for current in currents], loc=4)
        ax[i_curr].set_title(f'{curr} mA')
        ax[i_curr].set_xlabel('Distance (mm)')


        # remove the outer boxes
        for spine in ['top','bottom','right','left']:
            ax[i_curr].spines[spine].set_visible(False)
            # ax_time[i_chan].spines[spine].set_visible(False)
    

    ax[0].set_ylabel('Magnitude (uV)')
    fig.suptitle('Mean response minimum as a function of distance (per current level)')
    return fig


# ---------------------------------- #
def plot_LFP_min_times(resp_df:pd.DataFrame):
    '''
    plot the time for the minimum deviation as a function of current, per distance

    inputs: [default]
        resp_df: pd.DataFrame           - 

    outputs: 
        figure
    '''
    currents = resp_df.Current.unique()
    distances = resp_df.Distance.unique()
    currents.sort()
    distances.sort()

    fig,ax = plt.subplots(ncols=len(distances), sharex=True, sharey=True)

    for i_dist,dist in enumerate(distances):
        curr_cmp = resp_df.loc[resp_df.Distance == dist]
        ax[i_dist].scatter(curr_cmp.Current, curr_cmp.uMin_ts, s = 2, color='gray')
        curr_means = curr_cmp.groupby('Current').mean('uMin_ts')
        ax[i_dist].plot(curr_means.index, curr_means.uMin_ts, color='k')

        ax[i_dist].set_title(f'{dist} mm')
        ax[i_dist].set_xlabel('Current (mA)')

        # remove the outer boxes
        for spine in ['top','bottom','right','left']:
            ax[i_dist].spines[spine].set_visible(False)
    

    ax[0].set_ylabel('Time (ms)')
    fig.suptitle('Mean response minimum as a function of current (per distance)')

    return fig



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
def plot_mean_waveforms(spike_dict:dict, channel:int=0, fs:int=30000, ax:plt.axes = None, label:str = None, plot_std:bool = None):
    '''
    plot the mean threshold crossing for a channel, and a patch around the standard deviation.

    could theoretically look at splitting into different units

    inputs:
        spike_dict : dict           - spike dictionary
        channel : int               - channel number
        fs : int                    - sample rate (Hz) [30000]
        ax : plt.axes               - pyplot axes to plot on [None]
        label : str                 - label for the line [None]
        plot_std : bool             - should we plot the standard deviation [True]

    '''
    # create an axis if it doesn't exist
    if ax is None:
        fig,ax = plt.subplots()

    # go into the waveforms
    mean_wf = spike_dict[channel]['waveform'].mean(axis=0)

    # plot em
    ts = np.arange(len(mean_wf))/(fs/1000)
    line = ax.plot(ts, mean_wf, label=label) 
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Magnitude (uV)')

    # create std patch
    if plot_std:
        std_= spike_dict[channel]['waveform'].std(axis=0) # calculate the std
        # create the standard deviation patch
        std_array = np.ndarray((2*mean_wf.shape[0],2))
        std_array[:mean_wf.shape[0],:] = np.array(zip(mean_wf+std_, ts))
        std_array[mean_wf.shape[0]::-1,:] = np.array(zip(mean_wf-std_, ts))
        std_array[:,0] = std_array[:,0]/(fs/1000)
        std_patch = Polygon(std_array, alpha=0.2, color = line[0].get_color())
        ax.add_patch(std_patch)

# ---------------------------------- #
def plot_mean_FR(mean_FR:np.array, std_FR:np.array = None, channel:int = None, bin_ms:float = 1, ax:plt.axes = None, label:str = None):
    '''
    Plot the mean firing rates for a specific channel. 

    inputs:
        mean_FR : np.array          - mean firing rates
        std_FR : np.array           - standard deviation [None]
        channel : int               - channel to plot
        bin_ms  : float             - length of the bin (ms) [1]
        ax : matplotlib axes        - if we want to plot on an existing axis [None]
        label : str                 - label [None]

    outputs:
    '''
    if ax is None:
        fig, ax = plt.subplots()

    # create an array of timestamps
    ts = np.arange(mean_FR.shape[0])/bin_ms
    
    FR_line = ax.plot(ts, mean_FR[:,channel], label=label)
    # plot the standard deviations if desired
    if std_FR is not None:
        ts_std = np.concatenate([ts, ts[::-1]])
        std_patch = np.concatenate([mean_FR[:,channel] + std_FR[:,channel], mean_FR[::-1,channel] - std_FR[::-1,channel]])
        std_patch = Polygon(np.array(list(zip(ts_std,std_patch))), color=FR_line[0].get_color(), alpha=0.2)
        ax.add_patch(std_patch)

    ax.set_xlabel('Elapsed Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')



