import numpy as np
from scipy.optimize import curve_fit

from src.utils import model_quali_V

def fit_V(x, y, C, initial_guess=[0]):
    """
    Fit the A value of the model using the scaling law for the voltage
    """

    model = lambda x, A: model_quali_V(x, A, C)

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    popt, pcov = curve_fit(model, x, y, p0=initial_guess)

    return popt[0], pcov[0]