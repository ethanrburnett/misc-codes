# io_hw4_p3.py
#
# Use:
# Code for homework 4 problem 2 in ASTR 5800
# Computations of heat transfer in Io's interior, see comments below


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
from common_hw_functions import saveFigurePDF


def problem3_main():
    h = 0.8*1000.0  # m
    D = 110.0*1000.0  # m
    V = np.pi*(1./4.)*(D**2.)
    print('')
    print('Volume of volcano, m^3:')
    print(V)
    L = 40.0*1000.0  # Lithosphere thickness, m
    drho = 100.0  # kg/m^3
    g = 1.796  # m/s^2
    E = 1.0e11  # Pa
    w = (L**2.)*drho*g/E
    print('')
    print('Width of dike, meters:')
    print(w)
    eta = 1.0e3
    s = 10.0
    v = ((w**2.)/(2.0*eta))*drho*g
    Vdot = v*w*s
    time_form = V/Vdot
    print('')
    print('v (m/s), Vdot (m^3/s), time_form (years):')
    print(v)
    print(Vdot)
    print(time_form/(3600.0*24.0*365.0))
    kappa = 6.0e-7  # m^2/s
    tau_cond = (h**2.)/kappa
    print('conductive timescale:')
    print(tau_cond/(3600.0*24.0*365.0))

    delta = 0.8
    rho_c = 2500.0
    rho_m = 3000.0
    b = rho_c*delta/(rho_m - rho_c)
    print('')
    print('b:')
    print(b)
    return


# Execute:
problem3_main()
