from uncertainties import ufloat
from uncertainties.umath import *

from src.classes import SpecsParser

def get_specs(config):
    """
    Load the values from the specification file.
    """

    values = dict()
    specs = SpecsParser(f"specs/{config}.ini")
    
    print(f"\033[92mLoad parameters (SIU) from {config} specification file\033[0m")

    for (key, value) in specs.get_section('Values').items():
        print(f'{key:4} = {value:8} ± {float(specs.get_section("RelativeUncertainty")[key])*float(value):.2e}')
        values[key] = ufloat(float(value), float(specs.get_section('RelativeUncertainty')[key])*float(value))

    return values

def compute_A(values):
    """
    Calculate the value of the coefficient A from the specifications.
    """
    q31 = (values['d31'] + values['nu']*values['d33']) / ( (1-values['nu']) / values['ez'] - 2 * (values['nu'])**2 / values['ez'])

    A = values['ap'] * q31 * ( 3 * (1-values['nu']) / (7 - values['nu']) * values['rp'] / (values['l']*values['ez'])) ** (2/3) * values['gam']
    print(f'A    = {A.nominal_value:8.2e} ± {A.std_dev:.2e}')

    return {"nominal":A.nominal_value, "lower":A.nominal_value - A.std_dev, "upper":A.nominal_value + A.std_dev}

def compute_C(values):
    """
    Calculate the value of the coefficient C (capacitance) from the specifications.
    """
    C   = values['eps'] * 8.85e-12 * values['ae'] / values['l']
    print(f'C    = {C.nominal_value:8.2e} ± {C.std_dev:.2e}')

    return {"nominal":C.nominal_value, "lower":C.nominal_value - C.std_dev, "upper":C.nominal_value + C.std_dev}