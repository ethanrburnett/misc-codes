# differential_slope_hurst_hw3.py
#
# Use:
# Code for homework 3 problem 1 in ASTR 5800


# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np
import matplotlib.pyplot as plt
from common_hw_functions import build_sd_array, build_sd_array_v2, cluster_average_data, rms_erb, saveFigurePDF


def problem1_main():
    # location_path = '/Users/ethanburnett/Documents/University of Colorado Boulder/ASTR_5800(Fall_2019)/'
    file_name_1 = 'topo-data1.txt'  # Should be in venv-include!!
    file_name_2 = 'topo-data2.txt'
    data_array_1 = np.loadtxt(file_name_1)
    data_array_2 = np.loadtxt(file_name_2)
    # full resolution sd profiles for both data sets:
    sd_array_1a = build_sd_array(data_array_1)
    sd_array_2a = build_sd_array(data_array_2)
    # coarse resolution sd profiles for both data sets:
    sd_array_1b = build_sd_array(cluster_average_data(data_array_1, 10))  # Averaging appears correct!
    sd_array_2b = build_sd_array(cluster_average_data(data_array_2, 10))

    # Debug:
    average_data_check1 = cluster_average_data(data_array_1, 10)
    average_data_check2 = cluster_average_data(data_array_2, 10)
    check_avg_p1 = np.mean(data_array_1[0:10, 1])

    # RMS for each:
    sd_bar_1a = rms_erb(sd_array_1a[:, 1])
    sd_bar_1b = rms_erb(sd_array_1b[:, 1])
    sd_bar_2a = rms_erb(sd_array_2a[:, 1])
    sd_bar_2b = rms_erb(sd_array_2b[:, 1])
    H1 = (np.log(sd_bar_1b / sd_bar_1a) / np.log(9.0)) + 1.0
    H2 = (np.log(sd_bar_2b / sd_bar_2a) / np.log(9.0)) + 1.0

    # Debugging print stuff:
    print('sd_bar values 1a, 1b:')
    print(sd_bar_1a)
    print(sd_bar_1b)
    print('sd_bar values 2a, 2b:')
    print(sd_bar_2a)
    print(sd_bar_2b)
    print('Hurst exponents 1, 2:')
    print(H1)
    print(H2)

    # - - - - - - - - - PLOTS - - - - - - - - - -

    # USER: Set save options:
    file_path = '/Users/ethanburnett/Documents/University of Colorado Boulder/ASTR_5800_(Fall_2019)/HW3_ASTR5800/'
    fig_save_switch = 0  # 1 - save, 0 - don't save

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig1 = plt.figure(num=None, figsize=(5, 2.0), dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(data_array_1[:, 0], data_array_1[:, 1], label=r'profile 1', linewidth=1.0, color='black')
    ax1.plot(average_data_check1[:, 0], average_data_check1[:, 1], label=r'profile 1, averaged', linewidth=1.0, color='black', linestyle='dashed')
    ax1.set_xlabel(r'$x (m)$', fontsize=10)
    ax1.set_ylabel(r'Elevation (m)', fontsize=10)
    ax1.legend(prop={'size': 10})
    ax1.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig1 = plt.figure(num=None, figsize=(5, 2.0), dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(data_array_2[:, 0], data_array_2[:, 1], label=r'profile 2', linewidth=1.0, color='black')
    ax1.plot(average_data_check2[:, 0], average_data_check2[:, 1], label=r'profile 2, averaged', linewidth=1.0,
             color='black', linestyle='dashed')
    ax1.set_xlabel(r'$x (m)$', fontsize=10)
    ax1.set_ylabel(r'Elevation (m)', fontsize=10)
    ax1.legend(prop={'size': 10})
    ax1.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig1 = plt.figure(num=None, figsize=(5, 2.0), dpi=200)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(data_array_1[:, 0], data_array_1[:, 1], label=r'profile 1', linewidth=1.0, color='black')
    ax1.plot(data_array_2[:, 0], data_array_2[:, 1], label=r'profile 2', linewidth=1.0, color='black',  linestyle='dashed')
    ax1.set_xlabel(r'$x (m)$', fontsize=10)
    ax1.set_ylabel(r'Elevation (m)', fontsize=10)
    ax1.legend(prop={'size': 10})
    ax1.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    ax1.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
    if fig_save_switch == 1:
        figureName = 'hw3_p1_topo'
        saveFigurePDF(figureName, plt, file_path)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig2 = plt.figure(num=None, figsize=(5, 2.0), dpi=200)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(sd_array_1a[:, 0], sd_array_1a[:, 1], label=r'$s_{d,1}(L_{0})$', linewidth=1.0)
    ax2.plot(sd_array_1b[:, 0], sd_array_1b[:, 1], label=r'$s_{d,1}(L)$', linewidth=1.0)
    ax2.set_xlabel(r'$x$', fontsize=10)
    ax2.set_ylabel(r'$s_{d} = \frac{z_{3}-z_{2}}{x_{3}-x_{2}} - s^{*}$', fontsize=10)
    ax2.legend(prop={'size': 10})
    ax2.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.ylim(-0.26, 0.22)
    if fig_save_switch == 1:
        figureName = 'hw3_p1_sd1'
        saveFigurePDF(figureName, plt, file_path)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig3 = plt.figure(num=None, figsize=(5, 2.0), dpi=200)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(sd_array_2a[:, 0], sd_array_2a[:, 1], label=r'$s_{d,2}(L_{0})$', linewidth=1.0)
    ax3.plot(sd_array_2b[:, 0], sd_array_2b[:, 1], label=r'$s_{d,2}(L)$', linewidth=1.0)
    ax3.set_xlabel(r'$x$', fontsize=10)
    ax3.set_ylabel(r'$s_{d} = \frac{z_{3}-z_{2}}{x_{3}-x_{2}} - s^{*}$', fontsize=10)
    ax3.legend(prop={'size': 10})
    ax3.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.ylim(-0.26, 0.22)
    if fig_save_switch == 1:
        figureName = 'hw3_p1_sd2'
        saveFigurePDF(figureName, plt, file_path)

    plt.show()
    return


# Execute codes:
problem1_main()
