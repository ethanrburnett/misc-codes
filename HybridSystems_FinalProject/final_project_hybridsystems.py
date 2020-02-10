# final_project_hybridsystems.py
#
# Use:
#   For Hybrid Systems final project. Main project functions
#
# Requirements:
#   Packages: numpy, scipy, itertools, matplotlib
#   Codes: domain_functions, system_functions, project_trajectories
#
# Notes to user:
#   - Read all instructions in main
#   - Code is mostly generalized, but restricted to simple non-recursive LTL formulas

# INITIAL IMPORTING, PATHS
import numpy as np
import scipy
from scipy import spatial
from scipy import integrate
from scipy import optimize
from numpy import ndarray
import matplotlib.pyplot as plt
import itertools

# My functions:
from domain_functions import cost_fun_v1, nhat_tri_facet, cons_array_for_F, cons_array_for_stay, u_for_F_v1, \
    u_for_stay_v1, fg_for_domain
from system_functions import delta_trans, complete_R_array, getKeysByValue, obs_from_state, product_fcn, \
    which_Qp_state, which_T_state, unpack_delta_fwd, unpack_delta_back, check_bool, delta_prod, tree_build, \
    satisfying_paths_v2
from project_trajectories import fourier_coefficients, fourier_series, fourier_series_dot, fourier_series_ddot, \
    steering_de, integrator_stop_check

#  Transition system objects are explicitly redefined in this code for clarity:
class TransitionSystem(object):

    def __init__(self, X, Sigma, delta, Obs, o):
        self.X = X
        self.Sigma = Sigma  # Set to be length 4 (u op. 0, 1, 2, 3, j ! = 0: leave facet j). # Can change to generalize
        self.delta = delta
        self.Obs = Obs
        self.o = o


class Automaton(object):

    def __init__(self, Q, Q0, Sigma, R, F):
        self.Q = Q
        self.Q0 = Q0
        self.Sigma = Sigma
        self.R = R
        self.F = F


class ControlConstants(object):

    def __init__(self, A_sys, B_sys, K, signal_x_coeff, signal_y_coeff, signal_T):
        self.A_sys = A_sys
        self.B_sys = B_sys
        self.K = K
        self.signal_x_coeff = signal_x_coeff
        self.signal_y_coeff = signal_y_coeff
        self.signal_T = signal_T


def sharp_trajectory_steering(x0, t0, tf, h, c, B, traj_sched, traj_dur, tcheck, tri):
    # This function generates the un-smoothed trajectory using steering law dynamics
    # Initialize:
    x0_local = x0
    t0_local = t0
    t_out = t0
    state_out = np.array([[x0[0]], [x0[1]]])
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
    return t_out, state_out


def trajectory_smooth_fourier(t_out_prelim, state_out_prelim, n_max):
    t_out = np.append(t_out_prelim, t_out_prelim[0 : -1]+(2.*t_out_prelim[-1] - t_out_prelim[-2]))
    state_out = np.concatenate((state_out_prelim, np.fliplr(state_out_prelim[:, 0:-1])), axis=1)
    a_set1, b_set1 = fourier_coefficients(state_out[0, :], t_out, t_out[-1], n_max)
    a_set2, b_set2 = fourier_coefficients(state_out[1, :], t_out, t_out[-1], n_max)
    state_fit = np.zeros((2, len(t_out)))
    state_fit[0, :] = np.array([fourier_series(t, t_out[-1], a_set1, b_set1, n_max) for t in t_out])
    state_fit[1, :] = np.array([fourier_series(t, t_out[-1], a_set2, b_set2, n_max) for t in t_out])
    x_coeff = np.concatenate(([a_set1], [b_set1]), axis=0)
    y_coeff = np.concatenate(([a_set2], [b_set2]), axis=0)
    # Syntax note: n_max = x_coeff.shape[1] - 1
    return state_fit, x_coeff, y_coeff, t_out[-1]


def control_fcn(t, state, constants):
    K = constants.K
    T = constants.signal_T
    x_coeff = constants.signal_x_coeff
    y_coeff = constants.signal_y_coeff
    # Obtain signal to track at this time:
    r_steering = np.array([fourier_series(t, T, x_coeff[0, :], x_coeff[1, :], x_coeff.shape[1] - 1),
                           fourier_series(t, T, y_coeff[0, :], y_coeff[1, :], y_coeff.shape[1] - 1)])
    v_steering = np.array([fourier_series_dot(t, T, x_coeff[0, :], x_coeff[1, :], x_coeff.shape[1] - 1),
                           fourier_series_dot(t, T, y_coeff[0, :], y_coeff[1, :], y_coeff.shape[1] - 1)])
    vdot_steering = np.array([fourier_series_ddot(t, T, x_coeff[0, :], x_coeff[1, :], x_coeff.shape[1] - 1),
                              fourier_series_ddot(t, T, y_coeff[0, :], y_coeff[1, :], y_coeff.shape[1] - 1)])
    state_error = state - np.append(r_steering, v_steering)
    # Feedback with steering trajectory acceleration cancellation:
    u = -np.matmul(K, state_error) + vdot_steering
    return u


def system_de(t, state, constants):
    A = constants.A_sys
    B = constants.B_sys
    u = control_fcn(t, state, constants)
    state_rate = np.matmul(A, state) + np.matmul(B, u)
    return state_rate


def project_main():
    # - - - - - - - - - - - - - - - - - - - - - SECTION 1: DOMAIN DEFINITIONS - - - - - - - - - - - - - - - - - - - - -
    # User-specified nodes to form domain:
    # - List boundary nodes first, followed by inner nodes in any order
    nodes = np.array([[0., 0.], [0., 10.], [15., 10.], [15., 0.], [6., 10.], [9., 10.], [11., 0.], [2., 8.],
                      [8., 7.], [11., 2.], [5., 5.]])
    tri = spatial.Delaunay(nodes)

    # Optional display node, triangle data
    local_debug1 = 0
    if local_debug1 == 1:
        print ''
        print 'Entered points, triangles by vertex, and neighbors:'
        print tri.points  # Numbering of the points is in order of input
        print ''
        print tri.simplices  # From point numbers, give the triangles, starting with # 0
        print ''
        print tri.neighbors  # Give the triangle number of neighbors, and -1 for boundary

    tri_shape = tri.simplices.shape

    # - - - - - - - - - - - - - - - - - - - - - SECTION 2: SYSTEM DEFINITIONS - - - - - - - - - - - - - - - - - - - - -
    # USER-SET TRANSITION SYSTEM:
    # See class definition "TransitionSystem"
    user_Obs = ['R', 'S1', 'S2', 'K1', 'D']  # Define all possible observations in T
    user_o = {'R': 10, 'S1': 3, 'S2': 0, 'K1': 6, 'D': [1, 2, 4, 5, 7, 8, 9, 11, 12]}  # Defining o(x) for all x in T
    # Notes:
    # - user_o indexed as user_o['D'] for example, or user_o[user_Obs[k]] for chosen index k
    # - The observation 'D' defines permissible but otherwise uninteresting regions in the domain
    delta_array = delta_trans(tri).astype(int)  # Build the transition function "delta" for T
    T = TransitionSystem(np.arange(0, tri_shape[0]), np.array([0, 1, 2, 3]), delta_array, user_Obs, user_o)

    # USER-SET LTL-BASED AUTOMATON:
    # See class definition "Automaton" towards top of this file
    # Set all transition booleans: || Below, for LTL formula G(!K1)&F(S1 & F(R))
    f0 = lambda obs: int((obs != 'K1') and (obs != 'S1'))
    f1 = lambda obs: int((obs != 'K1') and (obs == 'S1') and (obs != 'R'))
    f2 = lambda obs: int((obs != 'K1') and (obs == 'S1') and (obs == 'R'))
    f3 = lambda obs: int((obs != 'K1') and (obs != 'R'))
    f4 = lambda obs: int((obs != 'K1') and (obs == 'R'))
    f5 = lambda obs: int(obs != 'K1')
    alph_fcns = [f0, f1, f2, f3, f4, f5]
    Sigma_A = np.arange(0, len(alph_fcns))  # Index representation

    # Set all states, specify initial and accepting:
    # Ordering: accepting states have highest numbers, initial state is zero.
    # Numbering should correlate to distance from initialization
    Q = np.array([0, 1, 2])  # k for q_k
    Q0 = Q[0]
    F = Q[2]
    # Non-trivial transitions, array representation: (completed automatically)
    # First entry: k for q_k, second entry: j for alph_fcns[j](), third entry: l for q_l after transition
    R_A = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 3, 1], [1, 4, 2], [2, 5, 2]])
    R_A = complete_R_array(R_A, len(Q), len(alph_fcns)).astype(int)
    A = Automaton(Q, Q0, Sigma_A, R_A, F)
    A.alph_fcns = alph_fcns

    # DEFINE PRODUCT AUTOMATON:
    # These lines immediately below need to be modified slightly for multi-valued A.Q0 and A.F:
    Qp = product_fcn(T.X, A.Q, len(Q)).astype(int)
    Q0p = product_fcn(T.X, [A.Q0], len(Q)).astype(int)
    Fp = product_fcn(T.X, [A.F], len(Q)).astype(int)

    # Build product transition function:
    delta_p = delta_prod(Qp, T.delta, A.R, A.Q, user_o, alph_fcns).astype(int)

    # USER-SET INITIAL CONDITIONS, FINAL SIMPLEX ID #:
    ICs = np.array([11.0, 9.5, 0.02, 0.0])  # These are x, y coordinates in domain, followed x_dot, y_dot
    xf = 10  # This is the simplex ID # of the final region

    # - - - - - - - - - - - - - - - - - - - - - SECTION 3: USER VISUALIZATION - - - - - - - - - - - - - - - - - - - - -
    # DISPLAY USER-INPUT SPECIFICATIONS FOR SYSTEM:
    # Print Options: (1 - print, 0 - don't print)
    print_system_def = 1
    print_system_trans = 0

    print ''
    if print_system_def == 1:
        print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
        print ''
        print 'BUILDING SYSTEMS FROM USER OPTIONS...'
        print ''
        if print_system_def == 0:
            print ''
            print 'NOTE: System definition print outputs suppressed by user'
        if print_system_trans == 0:
            print 'NOTE: Transition function print outputs suppressed by user'
        print ''
        print 'TRANSITION SYSTEM DEFINED:'
        print 'X:'
        print T.X
        print 'Sigma:'
        print T.Sigma
        print 'O:'
        print T.Obs
        print 'o:'
        print T.o
    if print_system_trans == 1:
        print 'delta:'
        print T.delta

    if print_system_def == 1:
        print ''
        print 'LTL-BASED SYSTEM DEFINED:'
        print 'Q:'
        print A.Q
        print 'Q0:'
        print A.Q0
        print 'Sigma indices:'
        print A.Sigma
        print 'F:'
        print A.F
    if print_system_trans == 1:
        print 'R (transition map for q, sigma):'
        print A.R

    if print_system_def == 1:
        print ''
        print 'PRODUCT AUTOMATON DEFINED: (Q_p, Q0_p, delta_p, F_p)'
        print 'Q_p: (#, x_idx, q_idx)'
        print Qp  # (Q_p index, x, q)
        print 'Q0_p:'
        print Q0p
        print 'F_p:'
        print Fp

    if print_system_trans == 1:
        print 'Transition function for product:'
        print delta_p
        print ''
    if print_system_def == 1:
        print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'

    # - - - - - - - - - - - - - - - - - - - SECTION 4: COMPUTE SATISFYING BEHAVIOR - - - - - - - - - - - - - - - - - - -
    print ''
    print 'COMPUTING SATISFYING BEHAVIOR...'
    print ''
    # GENERATE LTL-SATISFYING SYSTEM BEHAVIOR:
    x0 = tri.find_simplex(ICs[0:2])  # Compute which simplex contains initial conditions
    initial_prod_idx = which_Qp_state(x0, Q0, Q)  # Here we are assuming there is only 1 q0 and 1 qf (simple LTL)
    final_prod_idx = which_Qp_state(xf, F, Q)
    shortest_paths = 1  # Default value is 1

    sol_prod_states, sol_T_states, sol_trans = satisfying_paths_v2(initial_prod_idx, final_prod_idx, Qp, delta_p, shortest_paths)
    # Display results:
    print 'Satisfying state trajectory, in product automaton state IDs:'
    print sol_prod_states
    print ''
    print 'Satisfying order of states (simplices):'
    print sol_T_states
    print ''
    print 'Satisfying control index "j" for u_j:'
    print sol_trans
    print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'

    # - - - - - - - - - - - - - - - - - - SECTION 5: COMPUTE SATISFYING TRAJECTORIES - - - - - - - - - - - - - - - - - -
    print ''
    print 'COMPUTING SATISFYING STEERING LAW TRAJECTORY...'
    print ''

    # USER-SET DESIRED SOLUTION NUMBER:
    # (Row of satisfying simplex array)
    sol_idx = 1

    # USER-SET TRAJECTORY PARAMETERS:
    # Characteristic velocity and [B] matrix for steering law, and "loiter time":
    c = 0.05
    B = np.eye(2)  # Default is identity (see report)
    # The "loiter time" is the time spent controlling to remain inside simplex for control-to-stay (u0) option:
    traj_dur_time = 100.  # This value could be tuned based on "c" and a characteristic length, but that is overkill
    # Set dummy time array values for steering law based trajectory design:
    t0 = 0.0
    tf = 400.  # Another value that could be tuned. If steering law yields unscheduled transitions, increase this number
    tcheck = 10.0  # The default value for this is probably okay unless you really change characteristic velocity
    h = 0.1  # Usually leave this step value alone. It is explained in project_trajectories, in steering_trajectory_main
    # Smoothing parameter: number of coefficients in Fourier series
    n_coeff = 28  # If smoothed trajectory violates (unlikely for large n_coeff), increase this value as needed

    if sol_T_states.shape[0] >= sol_idx:
        sol_prod_select = sol_prod_states[sol_idx, :]
        sol_T_select = sol_T_states[sol_idx, :]
        sol_trans_select = sol_trans[sol_idx, :]
    else:
        sol_prod_select = sol_prod_states[0, :]  # Default to first listed solution
        sol_T_select = sol_T_states[0, :]
        sol_trans_select = sol_trans[0, :]

    # Convert selected solution arrays to traj_sched and traj_dur:
    control_opt_array = np.zeros((len(sol_trans_select),)).astype(int)
    facet_idx_array = np.zeros((len(sol_trans_select),)).astype(int)
    traj_dur = np.zeros((len(sol_trans_select),))
    for idx in range(len(sol_trans_select)):
        if sol_trans_select[idx] == 0:
            control_opt_array[idx] = 2
            facet_idx_array[idx] = 0
            traj_dur[idx] = traj_dur_time
        else:
            control_opt_array[idx] = 1
            facet_idx_array[idx] = sol_trans_select[idx] - 1
            traj_dur[idx] = 0.0
    traj_sched = np.concatenate(([sol_T_select], [control_opt_array], [facet_idx_array]), axis=0)

    t_out_sharp, state_out_sharp = sharp_trajectory_steering(ICs[0:2], t0, tf, h, c, B, traj_sched, traj_dur, tcheck, tri)
    state_out_smooth, x_coeff, y_coeff, T_smooth = trajectory_smooth_fourier(t_out_sharp, state_out_sharp, n_coeff)

    print 'Approximate satisfying trajectory travel time:'
    print t_out_sharp[-1].astype(int)
    print ''
    print 'Number of frequencies used in smoothed trajectory computation:'
    print n_coeff
    print ''
    print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'

    # Plot results:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.triplot(nodes[:, 0], nodes[:, 1], tri.simplices.copy())
    plt.plot(nodes[:, 0], nodes[:, 1], 'o')
    plt.plot(state_out_sharp[0, :], state_out_sharp[1, :], color='black', linestyle='dashed')
    plt.title('Partitioned Domain with Satisfying Steering Law Path')
    plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_out_sharp, state_out_sharp[0, :], color='black', linestyle='dashed', label=r'$x_{p}(t)$')
    ax.plot(t_out_sharp, state_out_sharp[1, :], color='black', label=r'$y_{p}(t)$')
    ax.plot(t_out_sharp, state_out_smooth[0, 0:len(t_out_sharp)], label=r'$x_{s}(t)$')
    ax.plot(t_out_sharp, state_out_smooth[1, 0:len(t_out_sharp)], label=r'$y_{s}(t)$')
    plt.title('Initial Planned and Smoothed Steering Law Trajectory $x(t), y(t)$', fontsize=12)
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
    plt.plot(state_out_smooth[0, 0:len(t_out_sharp)], state_out_smooth[1, 0:len(t_out_sharp)], color='black',
             linestyle='dashed')
    plt.title('Partitioned Domain with Smoothed Steering Law Path')
    plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    # - - - - - - - - - - - - SECTION 5: TRACKING THE STEERING LAW WITH TRUE DYNAMICAL SYSTEM - - - - - - - - - - - - -
    print ''
    print 'SIMULATING TRUE SYSTEM TRAJECTORY WITH TRACKING CONTROL...'
    print ''

    # Build A matrix for simple second-order system:
    A_sys = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    B_sys = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Build K matrix:
    tau = 5.0  # Settling time
    K = np.zeros((2, 4))
    # Desire critically damped closed-loop behavior:
    K[1, 3] = 2.0/tau
    K[0, 2] = 2.0/tau
    K[0, 0] = (K[0, 2] ** 2.) / 4.
    K[1, 1] = (K[1, 3] ** 2.) / 4.
    constants = ControlConstants(A_sys, B_sys, K, x_coeff, y_coeff, T_smooth)

    # Integrate the dynamics:
    fun2 = lambda t, x: system_de(t, x, constants)
    sol2 = integrate.solve_ivp(fun2, [0, t_out_sharp[-1]], ICs, method='BDF', t_eval=t_out_sharp, rtol=1e-13,atol=1e-13)
    rv_list = sol2.y

    # Reproduce control effort:
    control_list = np.zeros((2,len(t_out_sharp)))
    for idx in range(len(t_out_sharp)):
        control_list[:, idx] = control_fcn(t_out_sharp[idx], rv_list[:, idx], constants)

    # Plot all results:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_out_sharp, state_out_smooth[0, 0:len(t_out_sharp)], color='black', linestyle='dashed', linewidth=2.5,
            label=r'$x_{s}(t)$')
    ax.plot(t_out_sharp, state_out_smooth[1, 0:len(t_out_sharp)], color='black', linewidth=2.5, label=r'$y_{s}(t)$')
    ax.plot(t_out_sharp, rv_list[0, 0:len(t_out_sharp)], label=r'$x(t)$')
    ax.plot(t_out_sharp, rv_list[1, 0:len(t_out_sharp)], label=r'$y(t)$')
    plt.title('Steering Law Trajectory and True States $x(t), y(t)$', fontsize=12)
    ax.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.legend(prop={'size': 12})
    ax.tick_params(labelsize=12)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_out_sharp, control_list[0, 0:len(t_out_sharp)], label=r'$u_{1}(t)$')
    ax.plot(t_out_sharp, control_list[1, 0:len(t_out_sharp)], label=r'$u_{2}(t)$')
    plt.title('Control Signal', fontsize=12)
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
    plt.plot(state_out_smooth[0, 0:len(t_out_sharp)], state_out_smooth[1, 0:len(t_out_sharp)], color='black',
             linestyle='dashed', linewidth=2.0)
    plt.plot(rv_list[0, 0:len(t_out_sharp)], rv_list[1, 0:len(t_out_sharp)], linewidth=1.0)
    plt.title('Partitioned Domain with Planned Path and True Trajectory')
    plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    plt.show()

    return


project_main()
