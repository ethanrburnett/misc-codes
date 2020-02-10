# project_trajectories.py
#
# Use:
#   For Hybrid Systems final project. Set of functions to generate trajectories in domain
#
# Requirements:
#   Packages: numpy, scipy, itertools, matplotlib
#   Codes: domain_functions
#
# Notes:
#   [04/27/2019]: Verified single control-to-facet works with steering dynamics, as expected
#   [04/27/2019]: Developed, tested trajectory design with steering dynamics.
#   [04/27/2019]: Bug: Stay-inside doesn't always work for all h!

# INITIAL IMPORTING, PATHS
import numpy as np
import scipy
from scipy import spatial
from scipy import integrate
import matplotlib.pyplot as plt
import itertools
from domain_functions import cost_fun_v1, nhat_tri_facet, cons_array_for_F, cons_array_for_stay, u_for_F_v1, \
    u_for_stay_v1, fg_for_domain


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


def fourier_series_dot(data_time, T, a_set, b_set, n_max):
    f = np.array([(a_set[k]*((2*np.pi*k/T)) * np.sin(-2.*np.pi*k*data_time/T) +
                   b_set[k]*((2*np.pi*k/T))*np.cos(2.*np.pi*k*data_time/T)) for k in range(1, n_max+1)])
    return f.sum()


def fourier_series_ddot(data_time, T, a_set, b_set, n_max):
    f = np.array([(a_set[k]*((2*np.pi*k/T)**2.)*np.cos(2.*np.pi*k*data_time/T) +
                   b_set[k]*((2*np.pi*k/T)**2.)*np.sin(2.*np.pi*k*data_time/T)) for k in range(1, n_max+1)])
    return -f.sum()


def steering_de(t, x, fg_here):
    F = fg_here[0:2, 0:2]
    g = fg_here[0:2, 2]
    # print 'debug integrator AGAIN:'
    # print F
    # print g
    x_dot = np.matmul(F, x) + g
    return x_dot


def integrator_stop_check(h, x0, t0, dt_check, tmax, fg_here, tri, steering_de):
    # This function simulates the steering law trajectory in a simplex until stopping condition is met
    # Stopping options:
    # - Stop when simplex is left
    # - Stop when time is up (t > tmax)
    # Set master simulation time based on input options:
    N = int(((tmax - t0) / h) + 1.)
    master_time = np.linspace(t0, tmax, N)
    # Initialize local windows:
    num_eval = np.floor((tmax - t0) / dt_check)  # Number of evaluation windows
    N_window = np.int(np.rint(dt_check/h))  # (indices per window) - 1
    # Indices:
    window_count = 1  # Start in first time window
    local_start_idx = 0  # Start at t0
    stop_idx = -1  # Initial dummy value for stop_idx
    stop_cond = 0  # Stop_cond "off"
    idx_sf = 2  # Domain crossing-time overlap safety factor (avoid unstable numerical issues)
    # Determine initial triangle ID number:
    num0 = tri.find_simplex(x0)
    # Initialize state_out, time_out:
    t_out = t0
    state_out = np.array([[x0[0]], [x0[1]]])
    # Integrator function:
    fun_int = lambda t, x: steering_de(t, x, fg_here)
    # Integration:
    while stop_cond == 0:
        if window_count != num_eval:
            local_time_vec = master_time[local_start_idx : local_start_idx + N_window]
        else:
            local_time_vec = master_time[local_start_idx : N]
        # Round to nearest h:
        local_time_vec = np.around(local_time_vec, 2)  # Note this line needs to be commented/changed for non-default h
        # Compute, update values:
        sol_temp = integrate.solve_ivp(fun_int, [local_time_vec[0], local_time_vec[-1]], x0, method='BDF',
                                       t_eval=local_time_vec, rtol=1e-13, atol=1e-13)
        t_temp = sol_temp.t
        state_temp = sol_temp.y
        # Step:
        window_count = window_count + 1
        local_start_idx = local_start_idx + N_window - 1
        x0 = state_temp[:, -1]
        # Check for stopping condition (facet leaving):
        for idx in range(0, len(local_time_vec)):
            num_temp = tri.find_simplex(state_temp[:, idx])
            if (num_temp != num0) and (stop_idx == -1):
                if idx < (len(local_time_vec) - (idx_sf+1)):
                    stop_idx = idx + idx_sf
                else:
                    stop_idx = idx
                stop_cond = 1
        # Check for stopping condition (end-time-trigger):
        if local_start_idx > N:
            stop_cond = 1
        # Debug time indexing:
        if (len(t_temp) == N_window + 1) or (local_start_idx == N):
            stop_cond = 1
        # Update state_out:
        if stop_cond == 1:  # Final update
            state_out = np.concatenate((state_out, state_temp[:, 1:stop_idx]), axis=1)  # Don't double-count instances
            t_out = np.concatenate((t_out, t_temp[1:stop_idx]), axis=None)
        else:
            state_out = np.concatenate((state_out, state_temp[:, 1:len(t_temp)]), axis=1)
            # t_out = np.concatenate((t_out, t_temp[1:len(t_temp)]), axis=None)
            t_out = np.concatenate((t_out, t_temp[1:N_window]), axis=None)
    return t_out, state_out


def steering_trajectory_main():
    # To Do: Implement saving of critical crossing times!
    # Build domain from nodes:
    nodes = np.array([[0., 0.], [0., 10.], [15., 10.], [15., 0.], [6., 10.], [9., 10.], [11., 0.], [2., 8.],
                      [8., 7.], [11., 2.], [5., 5.]])
    tri = spatial.Delaunay(nodes)
    # Set initial conditions (position only entered here):
    # x0 = np.array([11.5, 0.2]) # Original
    # x0 = np.array([11.2, 0.2]) # First global
    # x0 = np.array([13.0, 0.2])  # Semi-final
    x0 = np.array([12.5, 0.6])  # Final
    # Set steering law schedule in traj_sched: (by row: 1 - tri ID num, 2 - control_opt, 3 - facet idx
    # control_opt: 1 - control to facet, 2 - stay inside
    # traj_sched = np.array([[10, 9, 8, 0, 0, 8, 9, 10, 10, 12, 4, 5, 7, 3, 1, 5, 4, 12, 10],
    #                        [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    #                        [1, 1, 1, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 1, 1, 2, 1, 0]])
    # # Set steering duration array (nonzero values for stay-inside only):
    # traj_dur = np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0,
    #                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.])
    # Trajectory that satisfies largest LTL specification:
    traj_sched = np.array([[10, 12, 4, 5, 1, 3, 3, 7, 5, 4, 12, 11, 8, 0, 0, 8, 9, 10],
                          [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2],
                          [2, 0, 0, 0, 2, 0, 2, 1, 1, 2, 2, 0, 1, 0, 0, 0, 2, 0]])
    traj_dur = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 100.])
    # traj_sched = np.array([[10, 9, 8, 0, 0, 8, 9, 10], [1, 1, 1, 2, 1, 1, 1, 2], [1, 1, 1, 0, 0, 0, 2, 0]])
    # traj_dur = np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 200.0])

    # Velocity scaling and [B] matrix:
    c = 0.05
    B = np.eye(2)

    # Set dummy time array values:
    t0 = 0.0
    tf = 400.0
    tcheck = 10.0
    h = 0.01  # If rounding to one digit in integrator_stop_check (smooth time), this should be 0.1
    # Initialize:
    x0_local = x0
    t0_local = t0
    t_out = t0
    state_out = np.array([[x0[0]], [x0[1]]])
    print ''
    print 'Generating satisfying trajectory...'
    for idx in range(0, traj_sched.shape[1]):  # traj_sched.shape[1]
        # Trajectory schedule data:
        num = traj_sched[0, idx]
        control_mode = traj_sched[1, idx]
        duration = traj_dur[idx]
        facet_id = traj_sched[2, idx]
        # Compute control:
        if control_mode == 1:
            u_optimal, fg_here = u_for_F_v1(tri, num, c, B, facet_id)
        else:
            u_optimal, fg_here = u_for_stay_v1(tri, num, c, B)
        # Simulate:
        if control_mode == 1:
            t_temp, state_temp = integrator_stop_check(h, x0_local, t0_local, tcheck, t0_local + tf, fg_here, tri, steering_de)
        else:
            t_temp, state_temp = integrator_stop_check(h, x0_local, t0_local, tcheck, t0_local + duration, fg_here, tri, steering_de)
        # Step:
        x0_local = state_temp[:, -1]
        t0_local = t_temp[-1]
        # Save values:
        t_out = np.concatenate((t_out, t_temp[1:len(t_temp)]), axis=None)
        state_out = np.concatenate((state_out, state_temp[:, 1:len(t_temp)]), axis=1)

    # Fourier-Series fit of trajectory:
    n_max = 24
    a_set1, b_set1 = fourier_coefficients(state_out[0, :], t_out, t_out[-1], n_max)
    a_set2, b_set2 = fourier_coefficients(state_out[1, :], t_out, t_out[-1], n_max)
    state_fit = np.zeros((2, len(t_out)))
    state_fit[0, :] = np.array([fourier_series(t, t_out[-1], a_set1, b_set1, n_max) for t in t_out])
    state_fit[1, :] = np.array([fourier_series(t, t_out[-1], a_set2, b_set2, n_max) for t in t_out])

    print ''
    print 'Total travel time (s):'
    print t_out[-1]

    # Plot results:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.triplot(nodes[:, 0], nodes[:, 1], tri.simplices.copy())
    plt.plot(nodes[:, 0], nodes[:, 1], 'o')
    plt.plot(state_out[0, :], state_out[1, :], color='black', linestyle='dashed')
    plt.title('Partitioned Domain with Satisfying Trajectory')
    plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    debug_plots = 0
    if debug_plots == 1:
        fig = plt.figure()
        plt.plot(np.linspace(t_out[0], t_out[-1], num = len(t_out)),
                 t_out - np.linspace(t_out[0], t_out[-1], num = len(t_out)), color='black', linestyle='dashed')
        plt.title('Simulation Time vs. Linear Time')
        plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_out, state_out[0, :], label=r'$x(t)$')
    ax.plot(t_out, state_out[1, :], label=r'$y(t)$')
    plt.title('Satisfying Trajectory $x(t), y(t)$', fontsize=12)
    ax.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.legend(prop={'size': 12})
    ax.tick_params(labelsize=12)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_out, state_out[0, :], color='black', linestyle='dashed', label=r'$x(t)$')
    ax.plot(t_out, state_out[1, :], color='black', label=r'$y(t)$')
    ax.plot(t_out, state_fit[0, :], label=r'$x_{s}(t)$')
    ax.plot(t_out, state_fit[1, :], label=r'$y_{s}(t)$')
    plt.title('Smoothed Trajectory $x(t), y(t)$', fontsize=12)
    ax.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.legend(prop={'size': 12})
    ax.tick_params(labelsize=12)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.triplot(nodes[:, 0], nodes[:, 1], tri.simplices.copy())
    plt.plot(nodes[:, 0], nodes[:, 1], 'o')
    plt.plot(state_fit[0, :], state_fit[1, :], color='black', linestyle='dashed')
    plt.title('Partitioned Domain with Smoothed Trajectory')
    plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    plt.show()
    return

# Uncomment to test:
# steering_trajectory_main()

