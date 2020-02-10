# heating_hw3.py
#
# Use:
# Code for homework 3 problem 3 in ASTR 5800


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
import matplotlib.pyplot as plt
from common_hw_functions import saveFigurePDF, rms_erb


def problem3_main():
    # All constants in meters, kg, K, etc:
    G = 6.674e-11
    M = 5.972e24
    R = 6371000.0
    cp = 1300.0
    t_now = 4.54e9
    deltaT_a = 3.0*G*M/(50.0*R*cp)
    print('')
    print('dT, part (a):')
    print(deltaT_a)

    f_vec = np.array([6.0e-8, 1.9e-8, 4.5e-7, 1.5e-7])
    th_vec = np.array([4.5e9, 0.7e9, 1.25e9, 14.0e9])
    sigma_vec = np.array([9.5e-5, 5.7e-4, 2.9e-5, 2.6e-5])
    qdot_vec_t0 = np.empty((len(sigma_vec,)))
    qdot_vec_tnow = np.empty((len(sigma_vec,)))
    for idx in range(len(qdot_vec_t0)):
        qdot_vec_t0[idx] = M*f_vec[idx]*sigma_vec[idx]
        qdot_vec_tnow[idx] = qdot_vec_t0[idx]*((0.5)**(t_now/th_vec[idx]))
    qdot_t0 = np.sum(qdot_vec_t0)
    qdot_tnow = np.sum(qdot_vec_tnow)
    print('')
    print('Individual heating rates at t0 then t_now, TW:')
    print(qdot_vec_t0/(1.0e12))
    print(qdot_vec_tnow/(1.0e12))
    print('Total heating rate at t0 then t_now, TW:')
    print(qdot_t0/(1.0e12))
    print(qdot_tnow/(1.0e12))

    t_melt = (4000.0 - 288.0)*M*cp/qdot_t0  # In seconds
    print('')
    print('Melt time:')
    print(t_melt/(3600.*24.*365.))

    phi_t0 = qdot_t0/(4.0*np.pi*(R**2.))
    phi_tnow = qdot_tnow/(4.0*np.pi*(R**2.))
    print('')
    print('Primordial and present-day flux, W/m^2:')
    print(phi_t0)
    print(phi_tnow)

    return


# Execute:
problem3_main()