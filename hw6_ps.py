# hw6_ps.py
#
# Use:
# Code for homework 6 problem 1 in ASTR 5800


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
# from common_hw_functions import saveFigurePDF


def hw6_main():
    r0 = 1815.0
    d = 30.0
    A0 = 4.*np.pi*(r0**2.)
    Af = 4.*np.pi*((r0 - d)**2.)
    epsilon_A = (Af - A0)/A0
    print('epsilon_A:')
    print(epsilon_A)
    epsilon_l = -d/r0
    print('epsilon_l:')
    print(epsilon_l)
    print('epsilon_A_approx:')
    print(2.*epsilon_l)
    E = 65.0*(1.0e9)
    v = 0.25
    sigma = -(E/(1. - v))*epsilon_l
    print('sigma:')
    print(sigma)
    # Part (c) of Problem 1:
    a_v = 2.5e-5
    Tm = 2200.0
    Ts = 110.0
    epsilon_th = a_v*(0.6*Tm - Ts)/3.
    print('epsilon_th:')
    print(epsilon_th)
    # Problem 2(a):
    lambda_b = 15.0  # km, fold wavelength
    alpha = lambda_b*(1000.)/(np.sqrt(2.)*np.pi)
    print('alpha:')
    print(alpha)
    E2 = 60.0*(1.0e9)
    g = 8.6
    rho_m = 3200.0
    t1 = ((3.*(alpha**4.)*(1. - (v**2.))*rho_m*g)/E2)**(1./3.)
    print('Thickness, km:')
    print(t1/1000.)
    D = (E2*(t1**3.))/(12.*(1. - (v**2.)))
    sigma_min = 4.*D/((alpha**2.)*t1)
    print('D:')
    print(D)
    print('sigma_min:')
    print(sigma_min)
    sigma_max = 0.1e9
    tbar = np.sqrt(sigma_max/(np.sqrt(rho_m*g*(E2/(3.*(1. - (v**2.)))))))
    print('tbar:')
    print(tbar)
    return

# Execute:
hw6_main()