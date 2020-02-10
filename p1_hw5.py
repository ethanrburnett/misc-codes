# p1_hw5.py
#
# Use:
# Code for homework 5 problem 1 in ASTR 5800
# Computations of maximum deviations for Phobos, and asteroid stuff


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
from common_hw_functions import saveFigurePDF


def main():
    G = 6.67408e-11 # m^3/kg-s^2...
    rho = 1900.0 #kg/m^3
    Rbar = 11.2*1.0e3  # m
    delta_h_obs = 2.8*1.0e3  # m

    # - - - - Calculate strength implied by surface irregularities: - - - -
    Y = (2./3.)*np.pi*G*(rho**2.)*Rbar*delta_h_obs  # Pascals
    print('')
    print('Strength implied by surface irregularities, Pa:')
    print(Y)

    # - - - - Calculate maximum delta-h for given Y value: - - - -
    Y_given = 10.0e6
    delta_h_hyp = 3.*Y_given/(2.*np.pi*G*(rho**2.)*Rbar)
    print('')
    print('For given strength, predicted height difference:')
    print(delta_h_hyp)

    # - - - - Estimated Delta-h for some asteroids - - - -
    mean_radius = np.array([49.0, 15.7, 16.84, 0.35, 0.245])
    phi_r = 30.0*(np.pi/180.0)
    f_f = np.tan(phi_r)
    delta_h = f_f*mean_radius
    print('')
    print(delta_h)

    # - - - - Problem 2 simple calculations - - - -
    a_enc = 252.0
    dphi1 = 45.0*(np.pi/180.0)
    dphi2 = 22.0*(np.pi/180.0)
    w = a_enc*np.array([dphi1, dphi2])
    print('')
    print('Wavelengths of features E1 and E4 (km):')
    print(w)

    E = 9.0e9
    t = 1.0e3
    v = 0.3
    rho_m = 1000.0
    g = 0.1
    alpha = (E*(t**3.)/(3.*(1.0 - (v**2.))*rho_m*g))**(1./4.)
    print('')
    print('Flexural parameter, problem 2(b):')
    print(alpha)
    return

# Execute:
main()
