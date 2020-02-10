# io_hw4.py
#
# Use:
# Code for homework 4 problem 2 in ASTR 5800
# Computations of heat transfer in Io's interior, see comments below


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
from common_hw_functions import saveFigurePDF, dx, newtons_method


class RayleighNumPhysConstants(object):

    def __init__(self, rho, g, alpha_v, Tm, Ts, L, eta0, A, kappa, k, a, H_star):
        self.rho = rho  # kg/m^3
        self.g = g  # m/s^2
        self.alpha_v = alpha_v  # 1/K
        self.Tm = Tm  # K
        self.Ts = Ts  # K
        self.L = L  # m
        self.eta0 = eta0  # kg/m-s
        self.A = A  # dimensionless
        self.kappa = kappa  # m^2/s
        self.k = k  # W/m-K
        self.a = a  # m
        self.H_star = H_star  # W/m^3, H value for conducted portion of heat transfer


def rayleigh_num(z, constants):
    # Extract, assign physical constants:
    rho = constants.rho
    g = constants.g
    alpha_v = constants.alpha_v
    H_star = constants.H_star
    k = constants.k
    kappa = constants.kappa
    L = constants.L
    a = constants.a
    Tm = constants.Tm
    Ts = constants.Ts
    eta0 = constants.eta0
    A = constants.A
    DT = -((H_star*a*z)/(3.0*k)) - ((H_star*(z**2.))/(6.0*k))
    T = Ts + DT
    eta = eta0*np.exp(A*(Tm/T))
    Ra = rho*g*alpha_v*DT*(L**3.)/(eta*kappa)
    return Ra


def d_rayleigh_num(z, constants):
    delta = 1.0  # meters
    dRadz = (rayleigh_num(z + delta, constants) - rayleigh_num(z - delta, constants))/(2.0*delta)
    return dRadz


def problem2_main():
    # - - - - - Part (a): Find total internal heat production H - - - - -
    q0 = 2.4  # Surface heat flow rate, W/m^2
    a = 1820.0  # radius, km
    H = 3.0*q0/(a*1000.0)  # W/m^3
    print('')
    print('H, W/m^3:')
    print(H)

    # - - - - - Part (b): Estimate melt depth of lithosphere in absence of volcanism - - - - -
    k = 2.0  # W/m-K
    Tm = 2000.0  # Melt temperature, K
    Ts = 110.0  # Surface temperature, K
    am = a*1000.0  #a, in meters
    z1 = a*(np.sqrt(1.0 - (6.0*k*(Tm-Ts)/(H*(am**2.))))-1.0)  # km
    print('')
    print('Melting depth for pure conduction (km):')
    print(-z1)

    # - - - Part (c): Estimate + analyze heat transport due to conduction - - -
    z_obs = -50.0
    H_star = 6.0*k*(Tm-Ts)/((a**2.)*(1.0e6*(1.0 - (((z_obs/a)+1.0)**2.))))  # W/m^3
    print('')
    print('H_star/H:')
    print(H_star/H)

    # Conducted heat flux:
    qcond = H_star*(a*1000./3.0)  # W/m^2
    print('')
    print('Conducted heat flux, q_cond, W/m^2:')
    print(qcond)

    # - - - - - Part (d): Find depth for critical Rayleigh number - - - - -

    # Define constants in base units:
    rho = 3530.0  # kg/m^3
    # # Compute density from mass:
    # rho = 8.93e22/(np.pi*(4.0/3.0)*(am**3.))
    # print('Density of Io, kg/m^3:')
    # print(rho)
    g = 1.796  # m/s^2
    alpha_v = 2.0e-5  # 1/K
    L = am/2.0  # m
    eta0 = 3400.0  # Pa
    A = 29.0
    kappa = 6.0e-7  # m^2/s
    constants = RayleighNumPhysConstants(rho, g, alpha_v, Tm, Ts, L, eta0, A, kappa, k, am, H_star)

    # Critical Rayleigh number and initial depth guess, z<0:
    rayleigh_crit = 1000.0  # Critical Rayleigh number for convection
    z_guess = -20.0*1000.0  # Guess of depth satisfying R_crit
    Ra_test = rayleigh_num(z_guess, constants)
    # print('')
    # print('Test Rayleigh Number at initial guess of melting depth:')
    # print(Ra_test)

    # Newton-Raphson solving for depth yielding critical Rayleigh number:
    f_local = lambda z: rayleigh_num(z, constants) - rayleigh_crit
    df_local = lambda z: d_rayleigh_num(z, constants)
    z_solve = newtons_method(f_local, df_local, z_guess, 1.0e-1)
    print('')
    print('Computed melt depth, km:')
    print(np.round(-z_solve/1000.0,2))
    return

# Execute:
problem2_main()