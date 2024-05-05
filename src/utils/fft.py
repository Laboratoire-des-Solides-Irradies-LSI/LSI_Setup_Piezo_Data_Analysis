import numpy as np

def get_fourier(trace):
    """Compute the Fourier transform of a trace."""
    return 2/len(trace) * np.abs(np.fft.rfft(trace))

def get_frequency(time):
    """Compute the frequency of a signal."""
    return np.fft.rfftfreq(time.size, d=time[1]-time[0])