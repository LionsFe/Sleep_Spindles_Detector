"""
- Author: Federico Lionetti
- Course: Biosignal Processing
- Faculty & Degree: UAS TECHNIKUM WIEN - Master's Degree Program: Medical Engineering & eHealth
- Date: 02 Nov. 2021


Hereby the author  present you a "Sleep Spindle" Detector that from raw EEGs files (the file format is .edf) finds,
highlights and then visually presents the sleep spindles that it can detect.
To do this, the code uses 5 parameters: 1) Absolute Sigma Power; 2) Relative Sigma Power; 3) Moving Correlation;
4) Moving Root-Mean-Square; 5) Moving Covariance.

In creating this code, the author relied on public codes of others sleep spindle detectors (i.e., YASA and Wonambi)
Please refer to these links for checking out the above-mentioned sources:
- https://raphaelvallat.com/yasa/build/html/index.html#
- https://wonambi-python.github.io/introduction.html

#######################################################################################################################

How the function 'detect_spindles' works
    - Input:
            eeg: Contains the input EEG signal as column vector
            fs: Sampling rate of the EEG signal
    - Output:
            spindles: Contains an n-by-2 matrix where each row corresponds to a detected sleep spindle and the first
            column contains the start time of spindle in seconds from the start of the recording and the second column
            contains the duration of the spindle in seconds.
"""

import mne
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.filter import filter_data
from scipy import signal
from scipy import stats
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d


def detect_spindles(eeg, fs):
    start = time.time()

    '''
    PARAMETERS THAT NEED TO BE CHANGED FOR STATISTICAL ANALYSIS AND CODE VALIDATION
    '''

    ABSOLUTE_SIGMA_POWER_THRESHOLD = 1.25
    RELATIVE_SIGMA_POWER_THRESHOLD = 0.20
    MOVING_CORRELATION_THRESHOLD = 0.69
    MOVING_RMS_THRESHOLD = 1.5
    COVARIANCE_THRESHOLD = 1.5

    THRESHOLD_FOR_GOOD_STUFF = 3.33
    DISTANCE_BETWEEN_SPINDLES = 500
    MAX_TIME_FOR_SPINDLES = 2
    MIN_TIME_FOR_SPINDLES = 0.5

    ####################################################################################################################

    file_name = eeg
    data = mne.io.read_raw_edf(file_name)
    raw_data = (data.get_data()[0]) * 1000000  # We multiply the data by this number because our units are uV and not V

    sample_freq = fs  # Getting the Sample Frequency

    # If the Sample Frequency is lower than 100 Hz, then we resample the data with a frequency of 100Hz
    if sample_freq < 100:
        fixed_data = data.resample(100)
        raw_data = fixed_data.get_data()[0] * 1000000
        sample_freq = float(100)

    # Defining the Time vector
    timesA = np.arange(raw_data.size) / sample_freq

    # Preprocessing the raw_data
    freq_board = [1, 30]
    data_broad = filter_data(raw_data, sample_freq, freq_board[0], freq_board[1], method='fir', verbose=0)

    ####################################################################################################################

    # ABSOLUTE SIGMA POWER

    dat_sigma_w = data_broad.copy()
    N = 20  # N order for the filter
    nyquist = sample_freq / 2
    Wn = 11 / nyquist
    sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
    dat_sigma_w = signal.sosfiltfilt(sos, dat_sigma_w)
    Wn = 16 / nyquist
    sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
    dat_sigma_w = signal.sosfiltfilt(sos, dat_sigma_w)

    dur = 0.3
    halfdur = dur / 2
    total_dur = len(dat_sigma_w) / sample_freq
    last = len(dat_sigma_w) - 1
    step = 0.1

    len_out = int(len(dat_sigma_w) / (step * sample_freq))
    out = np.zeros(len_out)
    tt = np.zeros(len_out)

    for i, j in enumerate(np.arange(0, total_dur, step)):
        beg = max(0, int((j - halfdur) * sample_freq))
        end = min(last, int((j + halfdur) * sample_freq))
        tt[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
        out[i] = np.mean(np.square(dat_sigma_w[beg:end]))

    dat_det_w = out
    dat_det_w[dat_det_w <= 0] = 0.000000001
    abs_sig_pow = np.log10(dat_det_w)

    interop = interp1d(tt, abs_sig_pow, kind='cubic', bounds_error=False,
                       fill_value=0, assume_sorted=True)

    tt = np.arange(dat_sigma_w.size) / sample_freq
    abs_sig_pow = interop(tt)

    # Counting the number of Spindles using ABSOLUTE SIGMA POWER

    text = 'spindles'
    spindles_counter_method0 = {}
    name = 0
    for item in abs_sig_pow:
        if item >= ABSOLUTE_SIGMA_POWER_THRESHOLD:
            spindles_counter_method0['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_counter_method0) == 1:
        text = 'spindle'
    print('Using ABSOLUTE SIGMA POWER we found', len(spindles_counter_method0), text)

    ###################################################################################################################

    # RELATIVE POWER IN THE SIGMA BAND

    # Computing the point-wise relative power using the Short-Term Fourier Transform
    f, t, SXX = signal.stft(data_broad, sample_freq, nperseg=(2 * sample_freq),
                            noverlap=((2 * sample_freq) - (0.2 * sample_freq)))

    # Keeping only the frequency of interest and Interpolating
    idx_band = np.logical_and(f >= freq_board[0], f <= freq_board[1])
    f = f[idx_band]
    SXX = SXX[idx_band, :]
    SXX = np.square(np.abs(SXX))
    PIPPO = RectBivariateSpline(f, t, SXX)
    t = np.arange(data_broad.size) / sample_freq
    SXX = PIPPO(f, t)
    sum_pow = SXX.sum(0).reshape(1, -1)
    np.divide(SXX, sum_pow, out=SXX)

    # Extracting the relative power in the sigma band
    idx_sigma = np.logical_and(f >= 11, f <= 16)
    rel_power = SXX[idx_sigma].sum(0)

    # Counting the number of Spindles with Relative Sigma Power
    text = 'spindles'
    spindles_counter_method1 = {}
    name = 0
    for item in rel_power:
        if item >= RELATIVE_SIGMA_POWER_THRESHOLD:
            spindles_counter_method1['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_counter_method1) == 1:
        text = 'spindle'

    print('Using RELATIVE SIGMA POWER, we found', len(spindles_counter_method1), text)

    ####################################################################################################################

    # MOVING CORRELATION

    data_sigma = data_broad.copy()
    N = 20  # N order for the filter
    nyquist = sample_freq / 2
    Wn = 11 / nyquist
    sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
    data_sigma = signal.sosfiltfilt(sos, data_sigma)

    Wn = 16 / nyquist
    sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
    data_sigma = signal.sosfiltfilt(sos, data_sigma)

    # Defining the moving sliding window and the step that are going to be used
    dur = 0.3
    halfdur = dur / 2
    total_dur = len(data_sigma) / sample_freq
    last = len(data_sigma) - 1
    step = 0.1

    len_out = int(len(data_sigma) / (step * sample_freq))
    out = np.zeros(len_out)
    tt = np.zeros(len_out)

    # Person Correlation Coefficient
    for i, j in enumerate(np.arange(0, total_dur, step)):
        beg = int(max(0, ((j - halfdur) * sample_freq)))  # if 'beg' is negative, then it goes to zero
        end = int(min(last, ((j + halfdur) * sample_freq)))  # if 'end' is greater than 'last', then it becomes 'last'
        tt[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
        win1 = data_sigma[beg:end]
        win2 = data_broad[beg:end]
        out[i] = stats.pearsonr(win1, win2)[0]

    moving_correlation = out
    # Let's interpolate
    interop = interp1d(tt, moving_correlation, kind='cubic', bounds_error=False,
                       fill_value=0, assume_sorted=True)

    tt = np.arange(data_sigma.size) / sample_freq

    moving_correlation = interop(tt)

    # Counting the number of Spindles with Moving Correlation
    spindles_counter_method2 = {}
    name = 0
    for item in out:
        if item >= MOVING_CORRELATION_THRESHOLD:
            spindles_counter_method2['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_counter_method2) == 1:
        text = 'spindle'
    print('Using MOVING CORRELATION, we found', len(spindles_counter_method2), text)

    ####################################################################################################################

    # MOVING ROOT-MEAN-SQUARE

    tt2 = np.zeros(len_out)
    moving_rms_final = np.zeros(len_out)

    # Creating a function for computing the Root Mean Square (RMS)
    def _rms(x):
        n = x.size
        ms = 0
        for iii in range(n):
            ms += x[iii] ** 2
        ms /= n
        return np.sqrt(ms)

    # Moving RMS
    for i, j in enumerate(np.arange(0, total_dur, step)):
        beg = int(max(0, ((j - halfdur) * sample_freq)))  # if beg is negative, then it goes to zero
        end = int(min(last, ((j + halfdur) * sample_freq)))  # if end is greater than last, then it becomes last
        tt2[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
        win3 = data_sigma[beg:end]
        moving_rms_final[i] = _rms(win3)

    # Let's Interpolate
    interop1 = interp1d(tt2, moving_rms_final, kind='cubic', bounds_error=False,
                        fill_value=0, assume_sorted=True)

    tt2 = np.arange(data_sigma.size) / sample_freq

    moving_rms_final = interop1(tt2)

    # TODO (?)
    def trim_both_std(x, cut=0.10):
        x = np.asarray(x)
        n = x.shape[-1]
        lowercut = int(cut * n)
        uppercut = n - lowercut
        atmp = np.partition(x, (lowercut, uppercut - 1), axis=-1)
        sl = slice(lowercut, uppercut)
        return np.nanstd(atmp[..., sl], ddof=1, axis=-1)

    trimmed_std = trim_both_std(moving_rms_final, cut=0.025)
    thresh_rms = moving_rms_final.mean() + MOVING_RMS_THRESHOLD * trimmed_std

    # Counting the number of Spindles with Moving RMS
    spindles_counter_method3 = {}
    name = 0
    for item in moving_rms_final:
        if item >= thresh_rms:
            spindles_counter_method3['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_counter_method3) == 1:
        text = 'spindle'
    print('Using MOVING RMS, we found', len(spindles_counter_method3), text)

    ##############################################################################################################

    # MOVING COVARIANCE

    dat_sigma_w = data_broad.copy()
    N = 20  # N order for the filter
    nyquist = sample_freq / 2
    Wn = 11 / nyquist
    sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
    dat_sigma_w = signal.sosfiltfilt(sos, dat_sigma_w)
    Wn = 16 / nyquist
    sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
    dat_sigma_w = signal.sosfiltfilt(sos, dat_sigma_w)

    dat_broad_w = data_broad.copy()
    N = 5  # N order for the filter
    nyquist = sample_freq / 2
    # Highpass
    Wn = 0.3 / nyquist
    sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
    dat_broad_w = signal.sosfiltfilt(sos, dat_broad_w)
    # Lowpass
    Wn = 30 / nyquist
    sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
    dat_broad_w = signal.sosfiltfilt(sos, dat_broad_w)

    dur = 0.3
    halfdur = dur / 2
    total_dur = len(dat_sigma_w) / sample_freq
    last = len(dat_sigma_w) - 1
    step = 0.1
    len_out = int(len(dat_sigma_w) / (step * sample_freq))
    out = np.zeros(len_out)
    tt = np.zeros(len_out)

    # Moving Covariance
    for i, j in enumerate(np.arange(0, total_dur, step)):
        beg = max(0, int((j - halfdur) * sample_freq))
        end = min(last, int((j + halfdur) * sample_freq))
        tt[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
        win1 = dat_sigma_w[beg:end]
        win2 = dat_broad_w[beg:end]
        out[i] = np.mean((win1 - np.mean(win1)) * (win2 - np.mean(win2)))

    dat_covar = out
    interop = interp1d(tt, dat_covar, kind='cubic', bounds_error=False,
                       fill_value=0, assume_sorted=True)
    tt = np.arange(dat_sigma_w.size) / sample_freq
    dat_covar = interop(tt)

    dat_det = dat_covar.copy()
    dat_det[dat_det < 0] = 0.00000001  # negative covariances are discated
    dat_det = np.log10(dat_det + 1)  # add 1 to avoid -inf

    dur = 30
    halfdur = dur / 2
    total_dur = len(dat_det) / sample_freq
    last = len(dat_det) - 1
    step = 1 / sample_freq
    # step = 0.1
    len_out = len(dat_det)
    out = np.zeros(len_out)
    tt = np.zeros(len_out)

    pcl_range = (10, 90)
    lo = np.percentile(dat_det, pcl_range[0])
    hi = np.percentile(dat_det, pcl_range[1])

    if pcl_range is not None:
        lo = np.percentile(dat_det, pcl_range[0])
        hi = np.percentile(dat_det, pcl_range[1])

    for i, j in enumerate(np.arange(0, total_dur, step)):
        beg = round(max(0, ((j - halfdur) * sample_freq)))
        end = round(min(last, ((j + halfdur) * sample_freq)))
        tt[i] = (np.column_stack((beg, end)).mean(1) / sample_freq)
        windat = stddat = dat_det[beg:end]
        if pcl_range is not None:
            stddat = windat[np.logical_and(windat > lo, windat < hi)]
        out[i] = (dat_det[i] - np.mean(windat)) / np.std(stddat)

    interop = interp1d(tt, out, kind='cubic', bounds_error=False,
                       fill_value=0, assume_sorted=True)

    tt = np.arange(dat_sigma_w.size) / sample_freq
    moving_covariance_final = interop(tt)

    text = 'spindles'
    spindles_counter_method4 = {}
    name = 0
    for item in moving_covariance_final:
        if item >= COVARIANCE_THRESHOLD:
            spindles_counter_method4['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_counter_method4) == 1:
        text = 'spindle'

    print('Using MOVING COVARIANCE, we found', len(spindles_counter_method1), text)
    ##############################################################################################################

    # LAST STEP

    # Combining all the results to make the "Decision"
    idx_absolute = (abs_sig_pow >= 1.25).astype(int)
    idx_rel_pow = (rel_power >= 0.2).astype(int)
    idx_mcorr = (moving_correlation >= 0.69).astype(int)
    idx_mrms = (moving_rms_final >= thresh_rms).astype(int)
    idx_cova = (moving_covariance_final >= 1.3).astype(int)

    idx_sum = (idx_absolute + idx_rel_pow + idx_mcorr + idx_mrms + idx_cova).astype(int)
    w = int(0.1 * sample_freq)
    idx_sum = np.convolve(idx_sum, np.ones(w) / w, mode='same')

    # Counting the number of Spindles with this Fourth  Method
    spindles_counter_method5 = {}
    name = 0
    for sssd in idx_sum:
        if sssd > 3.33:
            spindles_counter_method5['item' + str(name)] = [sssd]
        else:
            name += 1
    if len(spindles_counter_method5) == 1:
        text = 'spindle'
    print('\nIn the EEG at the end we generally found', len(spindles_counter_method5), text, '\n')

    ####################################################################################################################

    # GETTING ONLY THE GOOD STUFF

    where_sp = np.where(idx_sum > THRESHOLD_FOR_GOOD_STUFF)[0]

    def merge(index, sf):
        min_distance = DISTANCE_BETWEEN_SPINDLES / 1000. * sf
        idx_diff = np.diff(index)
        condition = idx_diff > 1
        idx_distance = np.where(condition)[0]
        distance = idx_diff[condition]
        bad = idx_distance[np.where(distance < min_distance)[0]]
        # Fill gap between events separated with less than min_distance_ms
        if len(bad) > 0:
            fill = np.hstack([np.arange(index[j] + 1, index[j + 1])
                              for i, j in enumerate(bad)])
            f_index = np.sort(np.append(index, fill))
            return f_index
        else:
            return index

    # Merge TOGETHER

    where_sp = merge(where_sp, sample_freq)
    sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
    idx_start_end = np.array([[k[0], k[-1]] for k in sp]) / sample_freq
    sp_start, sp_end = idx_start_end.T
    sp_dur = sp_end - sp_start

    output = []
    for useless in range(len(sp_start)):
        if sp_dur[useless] > MAX_TIME_FOR_SPINDLES:
            continue
        elif sp_dur[useless] < MIN_TIME_FOR_SPINDLES:
            continue
        else:
            output.append([sp_start[useless], sp_end[useless], sp_dur[useless]])
    arr_out = np.array(output)

    Marco_Ross_time = []
    for useless in range(len(sp_start)):
        if sp_dur[useless] > 1.5:
            continue
        elif sp_dur[useless] < 0.5:
            continue
        else:
            Marco_Ross_time.append([sp_start[useless], sp_dur[useless]])
    spindles = np.array(Marco_Ross_time)

    # Steps for visualization of the results
    mask = data_broad.copy()
    mask2 = data_broad.copy()
    OVERSERIES = pd.Series(mask, timesA)
    OVERSERIES2 = pd.Series(mask2, timesA)

    # Let's cut off all the data that do not correspond to our detected Sleep Spindles (TIME-CONSTRAINED)
    for fake in range(len(arr_out)):
        if fake == 0:
            OVERSERIES[0:arr_out[fake][0]] = np.nan
        elif fake == (len(arr_out) - 1):
            OVERSERIES[arr_out[fake - 1][1]:arr_out[fake][0]] = np.nan
            OVERSERIES[arr_out[fake][1]:timesA.max()] = np.nan
            break
        else:
            OVERSERIES[arr_out[fake - 1][1]:arr_out[fake][0]] = np.nan

    # Let's cut off all the data that do not correspond to our detected Sleep Spindles (FREE VERSION)
    for fake in range(len(sp_start)):
        if fake == 0:
            OVERSERIES2[0:sp_start[0]] = np.nan
        elif fake == len(sp_start) - 1:
            OVERSERIES2[sp_end[fake - 1]:sp_start[fake]] = np.nan
            OVERSERIES2[sp_end[fake]:timesA.max()] = np.nan
            break
        else:
            OVERSERIES2[sp_end[fake - 1]:sp_start[fake]] = np.nan

    if len(arr_out) == 1:
        text = 'spindle'
    print('\nUsing the TIME constriction we found', len(arr_out), text, '\n')

    ####################################################################################################################

    # LET'S ADD THE ANNOTATIONS
    # if there are any
    presence_of_annotation1 = 0
    presence_of_annotation2 = 0
    try:
        expert1 = (data.get_data()[1])
        expert1 = np.array(expert1)
        maskmask1 = data_broad.copy()
        OVERSERIES_ANNOTATION1 = pd.Series(expert1, timesA)
        OVERSERIES_EXPERT1 = pd.Series(maskmask1, timesA)
        if fs >= 100:
            for index in timesA:
                if OVERSERIES_ANNOTATION1[index] == 0:
                    OVERSERIES_EXPERT1[index] = np.nan
        else:
            for index in timesA:
                if (OVERSERIES_ANNOTATION1[index] > -0.05) and (OVERSERIES_ANNOTATION1[index] < 0.5):
                    OVERSERIES_EXPERT1[index] = np.nan

        presence_of_annotation1 += 1
    except IndexError:
        pass

    try:
        expert2 = (data.get_data()[2])
        expert2 = np.array(expert2)
        maskmask2 = data_broad.copy()
        OVERSERIES_ANNOTATION2 = pd.Series(expert2, timesA)
        OVERSERIES_EXPERT2 = pd.Series(maskmask2, timesA)
        if fs >= 100:
            for index in timesA:
                if OVERSERIES_ANNOTATION2[index] == 0:
                    OVERSERIES_EXPERT2[index] = np.nan
        else:
            for index in timesA:
                if (OVERSERIES_ANNOTATION2[index] > -0.05) and (OVERSERIES_ANNOTATION2[index] < 0.5):
                    OVERSERIES_EXPERT2[index] = np.nan
        presence_of_annotation2 += 1
    except IndexError:
        pass

    test_overlap = data_broad.copy()
    OVERLAP = pd.Series(test_overlap, timesA)

    if presence_of_annotation1 and presence_of_annotation2 == 1:
        for overlap in timesA:
            if OVERSERIES_EXPERT1[overlap] == OVERSERIES_EXPERT2[overlap]:
                if OVERSERIES_EXPERT1[overlap] == np.nan:
                    OVERLAP[overlap] = np.nan
                else:
                    continue
            else:
                OVERLAP[overlap] = np.nan

        fede_copy = data_broad.copy()
        FEDES_CODE = pd.Series(fede_copy, timesA)

        for overlap_ours in timesA:
            if OVERSERIES[overlap_ours] == OVERLAP[overlap_ours]:
                if OVERLAP[overlap_ours] == np.nan:
                    FEDES_CODE[overlap_ours] = np.nan
                else:
                    continue
            else:
                FEDES_CODE[overlap_ours] = np.nan

    ####################################################################################################################

    # PLOTTING THE FINAL GRAPHS
    plt.suptitle('5-FACTOR SLEEP SPINDLES DETECTOR on ' + file_name, fontweight='bold')
    plt.style.use('seaborn')
    plt.subplot(3, 1, 1)
    plt.title('N° of Sleep Spindles: ' + str(len(arr_out)))
    plt.plot(timesA, data_broad)
    plt.plot(OVERSERIES2, 'orange', label='Spindles w/out restrictions')
    plt.plot(OVERSERIES, 'black', label=' 0.5 < Spindles < 1.5 ')
    if presence_of_annotation1 and presence_of_annotation2 == 1:
        plt.plot(FEDES_CODE, 'cyan', label='Algorithm agrees with experts')
    plt.xlim(timesA.min(), timesA.max())
    plt.legend()

    plt.style.use('seaborn')
    plt.subplot(3, 1, 2)
    plt.plot(timesA, data_broad)
    if presence_of_annotation1 == 1:
        plt.plot(OVERSERIES_EXPERT1, 'tomato', label='Expert 1')
    if presence_of_annotation2 == 1:
        plt.plot(OVERSERIES_EXPERT2, 'yellowgreen', label='Expert 2')
    if presence_of_annotation1 and presence_of_annotation2 == 1:
        plt.plot(OVERLAP, 'cyan', label='Both Experts Agree')
    plt.xlim(timesA.min(), timesA.max())
    plt.legend()

    plt.style.use('seaborn')
    plt.subplot(3, 1, 3)
    plt.plot(timesA, idx_sum, '.-', markersize=5)
    plt.fill_between(timesA, THRESHOLD_FOR_GOOD_STUFF, idx_sum, where=idx_sum > THRESHOLD_FOR_GOOD_STUFF, color='red',
                     alpha=.8)
    plt.axhline(THRESHOLD_FOR_GOOD_STUFF, ls=':', lw=2, color='red', label='Final threshold')
    plt.legend()
    plt.xlim(timesA.min(), timesA.max())
    end = time.time()
    print("Time Elapsed:, ", end - start)
    plt.show()
    return spindles


detect_spindles('excerpt1.prep.edf', 100)
# detect_spindles('excerpt2.prep.edf', 200)
# detect_spindles('excerpt3.prep.edf', 50)
# detect_spindles('excerpt4.prep.edf', 200)
# detect_spindles('excerpt5.prep.edf', 200)
# detect_spindles('excerpt6.prep.edf', 200)
# detect_spindles('excerpt7.prep.edf', 200)
# detect_spindles('excerpt8.prep.edf', 200)
