# Albedo_Data_1.py
#
# Use:
# Code for final project in ASTR 5800
# Analyzes the albedo data from Nelson's 1986 Icarus paper on Europa



# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from common_hw_functions import saveFigurePDF


def fourier_coefficients(true_data, data_time, T, n_max):
    a_set = np.empty(n_max+1,)
    b_set = np.empty(n_max+1,)
    a_set[0] = (2./T)*np.trapz(true_data, x=data_time)
    b_set[0] = 0.0
    for idx in range(1, n_max + 1):
        cos_temp = np.cos(2.*np.pi*idx*data_time/T)
        sin_temp = np.sin(2.*np.pi*idx*data_time/T)
        a_set[idx] = (2. / T) * np.trapz(true_data * cos_temp, x=data_time)
        b_set[idx] = (2. / T) * np.trapz(true_data * sin_temp, x=data_time)
    return a_set, b_set


def fourier_series(data_time, T, a_set, b_set, n_max):
    f = np.array([(a_set[k]*np.cos(2.*np.pi*k*data_time/T) +
                   b_set[k]*np.sin(2.*np.pi*k*data_time/T)) for k in range(1, n_max+1)])
    return a_set[0]/2. + f.sum()


def r_expr_f(rmax,r0,a,n_star,t,l):
    r_val = rmax/(((rmax/r0) - np.exp(-a*np.cos(l)/n_star))*np.exp(a*np.cos(l + n_star*t)/n_star))
    return r_val


def r_expr_new(b,a0,a1,b1,l,t):
    a = b*((a1**2.)+ (b1**2.))/b1
    r0 = (a0/2.) - ((a1**2.)+ (b1**2.))/b1
    n = (a1/b1)*b
    r = r0 + (a*((b**2.) + (n**2.) + b*n*np.cos(l) + (b**2.)*np.sin(l))/(b*((b**2.) + (n**2.)))) \
        - (a*np.exp(-b*t)*((b**2. + n**2. + b*(n*np.cos(l) + b*np.sin(l))*(1. - 0.5*(n**2.)*(t**2.))
                            + b*n*t*(b*np.cos(l) - n*np.sin(l)))/(b*((b**2.) + (n**2.)))))
    return r


def main():
    # - - - - IMPORT DATA - - - - - -
    file_name_1 = 'Albedo_vs_Longitude_ScavengedData_v2_copy.txt'  # Should be in venv-include!!
    data_array_1 = np.loadtxt(file_name_1)
    data_array_1[:,0] = 360.0 - np.flip(data_array_1[:, 0])  # switch to deg. E
    data_array_1[:,1] = np.flip(data_array_1[:,1])

    # - - - - DATA OPERATIONS - - - -
    # Add the mean of the data to the missing zero degrees, this is needed for accurate Fourier fit:
    y_mean = np.mean(data_array_1[:, 1])
    data_array_1_mod = np.vstack((np.array([0.0, y_mean]), data_array_1,
                                  np.array([360.0, y_mean])))

    # Switch to radians:
    data_array_1_mod_rad = data_array_1_mod
    data_array_1_mod_rad[:,0] = (np.pi/180.)*data_array_1_mod_rad[:,0]
    # data_array_1_mod_rad[:,0] = data_array_1_mod_rad[:,0] - np.pi/2.  # Phase shift

    # Fit the data:
    n_max = 1
    a_set1, b_set1 = fourier_coefficients(data_array_1_mod_rad[:,1], data_array_1_mod_rad[:,0], 2.*np.pi, n_max)
    print('')
    print('Fourier coefficient sets, a_set1, b_set1:')
    print(a_set1)
    print(b_set1)
    data_fit_fourier = np.zeros((len(data_array_1_mod_rad[:,0]),))
    for idx in range(len(data_array_1_mod_rad[:,0])):
        data_fit_fourier[idx] = fourier_series(data_array_1_mod_rad[idx,0], 2.*np.pi, a_set1, b_set1, n_max)

    # Compute the parameter n_star*t:
    param = 2.*(a_set1[1]/b_set1[1])
    t_val_pwyll = 1.0e6
    n_star_est = param/t_val_pwyll
    T_est = (2.*np.pi / n_star_est)
    print('')
    print('Parameter n_star*t:')
    print(param)
    print('Mean motion for asynchronous rotation, est. 1:')
    print(n_star_est)
    print('Period of asynchronous rotation, est. 1:')
    print(T_est)

    # - - - - - - Test analytic fit to obtain parameter "b" - - - - - -
    # Set variables and parameters for Pwyll crater:
    tp = 1.0e6
    l = 89.0*np.pi/180.
    a0 = a_set1[0]
    a1 = a_set1[1]
    b1 = b_set1[1]
    rp = 0.45
    r0 = (a0 / 2.) - ((a1 ** 2.) + (b1 ** 2.)) / b1
    print('')
    print('r0:')
    print(r0)

    # Numerical solver:
    # fun1 = lambda b: r_expr_new(b, a0, a1, b1, l, tp) - rp
    # b_est = spo.fsolve(fun1, 0.001,full_output=1)  # Numerical solver not working, solve manually below!
    # print('')
    # print('Estimated additive decay constant b:')
    # print(b_est)

    # User-solved via iteration:
    b_est = 0.000002  # Change this value until r_expr = rp
    # b_est = 0.0000008
    rp_est = r_expr_new(b_est, a0, a1, b1, l, tp)
    print('')
    print('rp:')
    print(rp)
    print('rp_est for given b_est:')
    print(rp_est)
    n_star_est_2 = (a1/b1)*b_est
    T_est_2 = (2.*np.pi / n_star_est_2)
    print('')
    print('Rate of asynchronous rotation, est. 2:')
    print(n_star_est_2)
    print('Period of asynchronous rotation, est. 2:')
    print(T_est_2)
    print('')
    print(a1/b1)


    # - - - - - - Plot the data - - - - -
    # USER: Set save options:
    file_path = '/Users/ethanburnett/Documents/University of Colorado Boulder/ASTR_5800_(Fall_2019)/Final_Project/'
    fig_save_switch = 0  # 1 - save, 0 - don't save

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig2 = plt.figure(num=None, figsize=(5, 2.5), dpi=200)
    ax2 = fig2.add_subplot(1, 1, 1)
    # ax2.plot(360.0 - np.flip(data_array_1[:, 0]), np.flip(data_array_1[:, 1]), linewidth=1.0)
    ax2.plot((180./np.pi)*data_array_1_mod[:, 0], data_array_1_mod[:, 1],  label=r'original data', linewidth=1.0)
    ax2.plot((180./np.pi)*data_array_1_mod[:, 0], data_fit_fourier,  label=r'Fourier fit', linewidth=1.0)
    # ax2.plot(data_array_1[:, 0], data_array_1[:, 1], linewidth=1.0)
    ax2.set_xlabel(r'Longitude (deg. E)', fontsize=10)
    ax2.set_ylabel(r'Albedo ratio', fontsize=10)
    ax2.legend(prop={'size': 10})
    ax2.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    if fig_save_switch == 1:
        figureName = 'albedo_plot1'
        saveFigurePDF(figureName, plt, file_path)

    # Show plots:
    plt.show()

    return

# Execute:
main()
