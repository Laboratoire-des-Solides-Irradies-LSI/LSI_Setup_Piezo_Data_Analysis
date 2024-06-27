import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

def low_pass_filter(signal, cutoff_frequency, fs, order=2):
    """
    Apply a low-pass filter to a signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def demodulate(t, signal, frequency, win_size=5):
    """
    Demodulate a signal at a given frequency.
    """
    cut_freq = frequency / 2

    signal = savgol_filter(signal, window_length=win_size, polyorder=3, deriv=0, delta=t[1] - t[0])

    result = (np.exp(1j*2*np.pi*frequency*t)) * (signal - np.mean(signal)) # + np.exp(-1j*2*np.pi*frequency*t)
    result = low_pass_filter(result, cutoff_frequency=cut_freq, fs=1/(t[1] - t[0]))

    return np.mean(np.abs(result))