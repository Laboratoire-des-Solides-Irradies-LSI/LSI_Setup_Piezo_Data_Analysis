from scipy.signal import savgol_filter

def smooth(x, y, window_length=21, polyorder=3):
    """
    Smooth a signal using a Savitzky-Golay filter
    """
    return            savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=0, delta=x[1] - x[0])

def smooth_derivative(x, y, window_length=21, polyorder=3):
    """
    Smooth the derivative of a signal using a Savitzky-Golay filter
    """
    return            savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=1, delta=x[1] - x[0])