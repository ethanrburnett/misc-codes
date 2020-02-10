# monolith_hw3.py
#
# Use:
# Code for homework 3 problem 2 in ASTR 5800


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
import matplotlib.pyplot as plt
from common_hw_functions import saveFigurePDF, rms_erb


def g_function_scalar(G, V, dp, xc, d, x):
    # Units: G (m^3/kg*s^2, V (m^3), dp (kg/m^3), xc (m), d (m), x (m)
    scale_factor = 100.0  # Because anomaly measurements are in cm/s^2 instead of m/s^2
    g_val = scale_factor*G*V*dp/(((x-xc)**2.) + (d**2.))
    return g_val


def f_partials(G, V, dp, xc, d, x):
    # Units: G (m^3/kg*s^20, V (m^3), dp (kg/m^3), xc (m), d (m), x (m)
    dfdV_val = dp*G/((d**2.) + ((x-xc)**2.))
    dfdxc_val = 2.0*dp*G*V*(x-xc)/(((d**2.) + ((x-xc)**2.))**2.)
    dfdd_val = -2.0*dp*G*V*d/(((d**2.) + ((x-xc)**2.))**2.)
    f_jac = 100.0*np.array([dfdV_val, dfdxc_val, dfdd_val])
    return f_jac


def build_H(G, V_star, dp, xc_star, d_star, x_vec):
    H_array = np.empty((len(x_vec), 3))
    for index in range(len(x_vec)):
        H_array[index, :] = f_partials(G, V_star, dp, xc_star, d_star, x_vec[index])
    return H_array


def build_H_v2(G, V_star, dp, xc_star, d_star, x_vec):
    H_array = np.empty((len(x_vec), 4))
    for index in range(len(x_vec)):
        H_array[index, 0:3] = f_partials(G, V_star, dp, xc_star, d_star, x_vec[index])
        H_array[index, 3] = 100.0
    return H_array


def build_g_array(G, V, dp, xc, d, x_vec):
    g_modeled_array = np.empty((len(x_vec), 2))
    for idx in range(len(x_vec)):
        g_modeled_array[idx, 0] = x_vec[idx]
        g_modeled_array[idx, 1] = g_function_scalar(G, V, dp, xc, d, x_vec[idx])
    return g_modeled_array


def compute_residuals(obs_array, g_modeled_array):
    resids_vec = np.empty((len(obs_array[:, 0]),))
    for idx in range(len(obs_array[:, 0])):
        resids_vec[idx] = obs_array[idx, 1] - g_modeled_array[idx, 1]
    return resids_vec


def problem2_main():
    # Gravitational constant:
    G = 6.674e-11
    # Define gravity anomaly data:
    mgal_data = np.array([[0.0, -0.146*(1.0e-6)], [45.0, 1.35*(1.0e-6)], [60.0, 2.29*(1.0e-6)],
                          [70.0, 5.57*(1.0e-6)], [77.0, 6.03*(1.0e-6)], [85.0, 3.34*(1.0e-6)],
                          [115.0, -0.030*(1.0e-6)], [145.0, -0.161*(1.0e-6)], [175.0, -0.244*(1.0e-6)]])
    # Density estimates of anomaly and background:
    rho_est = 8300.0
    rho_0 = 1800.0
    dp = rho_est - rho_0
    # - - - - - Compute initial estimates of estimated parameters: - - - - -
    max_idx = np.argmax(mgal_data[:, 1])
    xc_star = mgal_data[max_idx, 0]  # x yielding maximum anomaly is assumed to be center location, xc
    g_max = mgal_data[max_idx, 1]

    d_star = 15.0  # Estimate depth, d = x(1/2) - xc, where x(1/2) = 92, xc = 77 (can also automate)
    V_star =(0.01*(g_max*(d_star**2.)))/(G*dp)  # Compute initial estimate of volume
    H_star = V_star ** (1. / 3.)
    V_star_high = (0.01*((g_max + 0.5e-6)*(d_star**2.)))/(G*dp)
    V_star_low = (0.01 * ((g_max - 0.5e-6) * (d_star ** 2.))) / (G * dp)
    H_star_high = (V_star_high)**(1./3.)
    H_star_low = (V_star_low)**(1./3.)
    print('')
    print('Hypothetical errors in volume estimate V*, high and low:')
    print(V_star_high - V_star)
    print(V_star_low - V_star)
    print('Corresponding errors in H*, high and low:')
    print(H_star_high - H_star)
    print(H_star_low - H_star)

    # g_max_rep = G*V_star*dp/(d_star**2.)

    # - - - - - Build arrays for estimate correction: - - - - - -
    g_modeled_array = build_g_array(G, V_star, dp, xc_star, d_star, mgal_data[:, 0])
    delta_g_vec = compute_residuals(mgal_data, g_modeled_array)
    H_array = build_H_v2(G, V_star, dp, xc_star, d_star, mgal_data[:, 0])
    W_array = np.diagflat([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    inter_mat1 = np.matmul(np.transpose(H_array), np.matmul(W_array, H_array))
    inter_mat2 = np.linalg.inv(inter_mat1)
    inter_mat3 = np.matmul(inter_mat2, np.matmul(np.transpose(H_array), W_array))

    # Estimate of correction to volume, center location, and depth (with bias):
    x_hat = np.matmul(inter_mat3, delta_g_vec)
    x_hat[3] = x_hat[3]*100.0
    print('')
    print('x_hat: volume correction (m^3), xc correction (m), depth correction (m), background signal correction (Gal)')
    print(x_hat)

    # Update the estimates:
    V_new = V_star + x_hat[0]
    xc_new = xc_star + x_hat[1]
    d_new = d_star + x_hat[2]
    g_modeled_array2 = build_g_array(G, V_new, dp, xc_new, d_new, mgal_data[:, 0]) + x_hat[3]

    H_array_0 = build_H(G, V_star, dp, xc_star, d_star, mgal_data[:, 0])
    W_array_0 = np.diagflat([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    inter_mat1_0 = np.matmul(np.transpose(H_array_0), np.matmul(W_array_0, H_array_0))
    inter_mat2_0 = np.linalg.inv(inter_mat1_0)
    inter_mat3_0 = np.matmul(inter_mat2_0, np.matmul(np.transpose(H_array_0), W_array_0))

    # Estimate of correction to volume, center location, and depth (without bias):
    x_hat_0 = np.matmul(inter_mat3_0, delta_g_vec)
    V_new_0 = V_star + x_hat_0[0]
    xc_new_0 = xc_star + x_hat_0[1]
    d_new_0 = d_star + x_hat_0[2]
    g_modeled_array1 = build_g_array(G, V_new_0, dp, xc_new_0, d_new_0, mgal_data[:, 0])

    print('')
    print('- - - - - - - - - - - - - - - - - - - -')
    print('')
    print('Initial volume estimate (m^3):')
    print(V_star)
    print('Initial H estimate (m):')
    print(H_star)
    print('Initial center location estimate (m):')
    print(xc_star)
    print('Initial depth estimate (m):')
    print(d_star)

    print('')
    print('- - - no bias: - - -')
    print('Final volume estimate (m^3):')
    print(V_new_0)
    print('Final H estimate (m):')
    H_new_0 = V_new_0 ** (1. / 3.)
    print(H_new_0)
    print('Final center location estimate (m):')
    print(xc_new_0)
    print('Final depth estimate (m):')
    print(d_new_0)

    print('')
    print('- - - with bias: - - -')
    print('Final volume estimate (m^3):')
    print(V_new)
    print('Final H estimate (m):')
    H_new = V_new ** (1./3.)
    print(H_new)
    print('Final center location estimate (m):')
    print(xc_new)
    print('Final depth estimate (m):')
    print(d_new)

    # Compute RMS of residuals:
    delta_g_vec2 = compute_residuals(mgal_data, g_modeled_array2)
    delta_g_vec0 = compute_residuals(mgal_data, g_modeled_array1)
    rms_fit1 = rms_erb(delta_g_vec)
    rms_fit2 = rms_erb(delta_g_vec2)
    rms_fit0 = rms_erb(delta_g_vec0)

    # Print RMS of residuals:
    print('')
    print('RMS of residuals (fit 1, fit 2, fit 3):')
    print(rms_fit1)
    print(rms_fit0)
    print(rms_fit2)

    # Curve fits for plot:
    x_vector = np.linspace(0.0, 175.0, num=351)
    g_curve_1 = build_g_array(G, V_star, dp, xc_star, d_star, x_vector)
    g_curve_2 = build_g_array(G, V_new, dp, xc_new, d_new, x_vector) + x_hat[3]
    g_curve_0 = build_g_array(G, V_new_0, dp, xc_new_0, d_new_0, x_vector)

    # - - - - - PLOTS - - - - -

    # USER: Set save options:
    file_path = '/Users/ethanburnett/Documents/University of Colorado Boulder/ASTR_5800_(Fall_2019)/HW3_ASTR5800/'
    fig_save_switch = 0  # 1 - save, 0 - don't save

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig1 = plt.figure(num=None, figsize=(6, 2.8), dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.scatter(mgal_data[:, 0], (1.0e6)*mgal_data[:, 1], marker='.', color='black')
    ax1.plot(x_vector, (1.0e6)*g_curve_1[:, 1], label=r'fit 1: analytic', linewidth=1.0)
    ax1.plot(x_vector, (1.0e6)*g_curve_0[:, 1], label=r'fit 2: least-squares', linewidth=1.0)
    ax1.plot(x_vector, (1.0e6)*g_curve_2[:, 1], label=r'fit 3: least-squares w/ bias', linewidth=1.0)
    plt.gca().set_prop_cycle(None)
    ax1.set_xlabel(r'Distance (m)', fontsize=10)
    ax1.set_ylabel(r'$\Delta g$ ($\mu$Gal)', fontsize=10)
    ax1.legend(prop={'size': 8})
    ax1.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    if fig_save_switch == 1:
        figureName = 'hw3_p2_grav_v2'
        saveFigurePDF(figureName, plt, file_path)

    plt.show()
    return


# Execute:
problem2_main()