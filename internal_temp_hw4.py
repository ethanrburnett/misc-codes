# internal_temp_hw4.py
#
# Use:
# Code for homework 4 problem 1 in ASTR 5800


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
from common_hw_functions import saveFigurePDF, rms_erb


def problem1_main():
    radius_data = np.array([512.0, 134.0, 263.0, 15.0, 5.0])  # km
    mass_data = np.array([9.3835e20, 2.67e19, 2.59e20, 4.2e16, 6.687e15])  # kg
    volume_data = (4.0/3.0)*np.pi*np.power(radius_data, 3.*np.ones((5,)))  # km^3
    density_data = np.divide(mass_data, volume_data)  # kg/km^3
    k = 2000.0  # W/km-k
    Hp = 5.23e-12  # W/kg
    H_data = Hp*density_data  # W/km^3
    a2_data = np.power(radius_data, 2.*np.ones((5,)))  # km^2
    dT_data = (1.0/(6.0*k))*np.multiply(H_data, a2_data)
    # Debug:
    print('')
    print('radius, mass, volume, density:')
    print(radius_data)
    print(mass_data)
    print(volume_data)
    print(density_data)
    print('')
    print('Temperature differences, K:')
    print(dT_data)

    return


# Execute:
problem1_main()
