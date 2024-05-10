import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from .filter import smooth, smooth_derivative

from src.utils import get_fourier, get_frequency

def R_effective(R_decade_box):
    """
    Effective resistance taking into account the measusrement tool
    """
    return            1/(1e-6 + 1/R_decade_box)

def ode_system(t, V, dp, R, C, A):
    """
    ODE system for the Runge-Kutta solver
    """
    dVdt              = (-V - A*R*dp(t)) / (C*R)

    return            dVdt

def model_quant_V_freq(time, pressure, A, C, R_decade_box, filter_window=21):
    """
    Quantitative model for the voltage in the frequency domain
    """

    omega           = 2*np.pi*get_frequency(time)
    pressure_smooth = smooth(time, np.power(pressure, 2/3), window_length=filter_window)

    R               = R_effective(R_decade_box)
    
    return            A*omega*np.abs(R / (1j - omega*R*C)) * get_fourier(pressure_smooth)

def model_quant_V_trace(time, pressure, A, C, R_decade_box, filter_window=21):
    """
    Quantitative model for the voltage in the time domain
    """
    derivative      = smooth_derivative(time, np.power(pressure, 2/3), window_length=filter_window)

    derivative_int  = interp1d(time, derivative, kind='quadratic', fill_value="extrapolate")

    R               = R_effective(R_decade_box)
    
    return            solve_ivp(lambda t, V : ode_system(t, V, derivative_int, R, C, A), (time[0], time[-1]), [0], t_eval=time, method='RK45').y[0]

def model_quali_V(x, A, C):
    """
    Qualitative model for the voltage
    """
    return            A * np.abs(x / (1j - C * x))