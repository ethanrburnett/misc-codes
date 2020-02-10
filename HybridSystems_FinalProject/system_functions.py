# system_functions.py
#
# Use:
#   For Hybrid Systems final project. Set of functions for user to specify transition systems
#
# Requirements:
#   Packages: numpy, scipy, itertools, matplotlib
#   Codes: N/A

# INITIAL IMPORTING, PATHS
import numpy as np
from scipy import spatial


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


def delta_trans(tri):
    # From simplex data, assemble the data for the transition function "delta" for T
    # Structure: 4*N rows of: [triangle # before transition, control_ID, triangle # after transition]
    # control_ID = 0 (stay inside simplex), 3 >= j > 0 (leave facet j)
    # Note that for a transition out of the domain, the identifier "-1" is reused
    tri_shape = tri.simplices.shape
    delta_array = np.zeros((4*tri_shape[0], 3))
    for idx in range(tri_shape[0]):
        for idx2 in range(4):
            if idx2 != 0:
                delta_array[4 * idx + idx2, :] = np.array([idx, idx2, tri.neighbors[idx, idx2-1]])
            else:
                delta_array[4 * idx + idx2, :] = np.array([idx, idx2, idx])
    return delta_array


def complete_R_array(R_pre, lQ, lO):
    counter = 0
    max_count = R_pre.shape[0]
    R_array = np.zeros((lQ*lO, 3))
    for idx1 in range(lQ):
        for idx2 in range(lO):
            if (R_pre[counter, 0] == idx1) and (R_pre[counter, 1] == idx2) and (counter <= max_count):
                R_array[lO * idx1 + idx2, :] = R_pre[counter, :]
                counter = counter + 1
            else:
                R_array[lO * idx1 + idx2, :] = np.array([idx1, idx2, -1])
    return R_array


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


def obs_from_state(state_idx, obs_dict):
    if len(getKeysByValue(obs_dict, state_idx)) > 0:
        obs = getKeysByValue(obs_dict, state_idx)
    else:
        obs = 'D'  # In an otherwise unspecified but permissible region of the domain
    return obs[0]


def product_fcn(T_array, A_array, lQ):
    # Defines product arrays, for indexing product automaton entities
    lT = len(T_array)
    lA = len(A_array)
    P_array = np.zeros((lT*lA, 3))
    counter = 0
    for idx1 in range(lT):
        for idx2 in range(lA):
            P_array[lA * idx1 + idx2, :] = np.array([lQ * idx1 + A_array[idx2], T_array[idx1], A_array[idx2]])
            counter = counter + 1
    return P_array


def which_Qp_state(x_idx, q_idx, Q):
    lQ = len(Q)
    Qp_idx = lQ*x_idx + Q[q_idx]
    return Qp_idx


def which_T_state(Qp, sol_prod_states):
    T_state_out = np.zeros((sol_prod_states.shape[0], sol_prod_states.shape[1])).astype(int)
    for idx1 in range(sol_prod_states.shape[0]):
        for idx2 in range(sol_prod_states.shape[1]):
            T_state_out[idx1, idx2] = Qp[sol_prod_states[idx1, idx2], 1]
    return T_state_out


def unpack_delta_fwd(delta_array, n_states, n_trans, state_sel):
    # From a delta_array of row-form [state_idx, trans_idx, state_idx2 (or -1)], return allowable transition indices
    # Also return successors for allowable transitions
    trans_save = []
    fut_save = []
    for idx in range(n_states):
        for idx2 in range(n_trans):
            if (delta_array[n_trans*idx + idx2, 0] == state_sel) and (delta_array[n_trans*idx + idx2, 2] != -1):
                # print delta_array[n_trans*idx + idx2, 2]
                trans_save.extend([delta_array[n_trans*idx + idx2, 1]])
                fut_save.extend([delta_array[n_trans*idx + idx2, 2]])
    return trans_save, fut_save


def unpack_delta_back(delta_array, n_states, n_trans, state_sel):
    # From a delta_array of row-form [state_idx, trans_idx, state_idx2 (or -1)], return allowable transition indices
    # Also return predecessors for allowable transitions
    trans_save = []
    pred_save = []
    for idx in range(n_states):
        for idx2 in range(n_trans):
            if (delta_array[n_trans*idx + idx2, 2] == state_sel) and (delta_array[n_trans*idx + idx2, 2] != -1):
                # print delta_array[n_trans*idx + idx2, 2]
                trans_save.extend([delta_array[n_trans*idx + idx2, 1]])
                pred_save.extend([delta_array[n_trans*idx + idx2, 0]])
    return trans_save, pred_save


def check_bool(q_idx, J_array, R, alph_fcns, obs_res):
    # From given q_idx and available transitions (Boolean formulas), check if the input observation allows a transition
    q_res = -1
    for idx in J_array:
        # print 'should be Boolean:'
        # print alph_fcns[idx](obs_res)
        if alph_fcns[idx](obs_res):
            q_res = R[len(alph_fcns)*q_idx + idx, 2]
    return q_res


def delta_prod(Qp, delta, R, Q, user_o, alph_fcns):
    delta_p = np.zeros((4*Qp.shape[0], 3))
    for idx in range(Qp.shape[0]):
        for idx2 in range(4):
            x_idx_temp = Qp[idx, 1]
            # print 'x_temp, control_idx, q_temp:'
            # print x_idx_temp
            # print idx2
            d_idx_temp = 4*x_idx_temp + idx2
            x_res = delta[d_idx_temp, 2]
            q_idx_temp = Qp[idx, 2]
            # print q_idx_temp
            if x_res != -1:
                # print 'x_res:'
                # print x_res
                obs_res = obs_from_state(x_res.item(), user_o)
                J_array, fut_array = unpack_delta_fwd(R, len(Q), len(alph_fcns), q_idx_temp)  # 3, 6
                q_res = check_bool(q_idx_temp, J_array, R, alph_fcns, obs_res)
                if q_res != -1:
                    delta_p[4*idx + idx2, :] = np.array([Qp[idx, 0], idx2, which_Qp_state(x_res, q_res, Q)])
                else:
                    delta_p[4 * idx + idx2, :] = np.array([Qp[idx, 0], idx2, -1])
            else:
                delta_p[4*idx + idx2, :] = np.array([Qp[idx, 0], idx2, -1])
    return delta_p


def tree_build(future_states, future_trans, delta_p, Qp):
    future_states_out = np.empty((2, future_states.shape[1] + 1), int)
    future_trans_out = np.empty((2, future_states.shape[1] + 1), int)
    for idx in range(future_states.shape[0]):
        state_future_temp = future_states[idx, -1]
        trans_temp, pred_temp = unpack_delta_back(delta_p, Qp.shape[0], 4, state_future_temp)
        # print len(pred_temp)
        future_states_sub = np.zeros((len(pred_temp), future_states.shape[1] + 1)).astype(int)
        future_trans_sub = np.zeros((len(pred_temp), future_states.shape[1] + 1)).astype(int)
        if len(pred_temp) != 0:
            for idx2 in range(len(pred_temp)):
                # print future_states[idx, :].item()
                # print pred_temp[idx2]
                if len(future_states[idx, :]) == 1:
                    future_states_sub[idx2, :] = np.array([future_states[idx, :].item(), pred_temp[idx2]])
                    future_trans_sub[idx2, :] = np.array([future_trans[idx, :].item(), trans_temp[idx2]])
                else:
                    future_states_sub[idx2, :] = np.append(future_states[idx, :], [pred_temp[idx2]])
                    future_trans_sub[idx2, :] = np.append(future_trans[idx, :], [trans_temp[idx2]])
            # print 'state array and transition array for this future:'
            # print future_states_sub
            # print future_trans_sub
            future_states_out = np.concatenate((future_states_out, future_states_sub), axis=0)
            future_trans_out = np.concatenate((future_trans_out, future_trans_sub), axis=0)
    # Delete dummy data:
    future_states_out = np.delete(future_states_out, [0, 1], 0)
    future_trans_out = np.delete(future_trans_out, [0, 1], 0)
    return future_states_out, future_trans_out


def satisfying_paths(final_prod_idx, max_iter, Qp, delta_p, recurs_cut):
    # Build the initial entries in array of states, + associated transitions that bring you from them to final state:
    trans_init, pred_init = unpack_delta_back(delta_p, Qp.shape[0], 4, final_prod_idx)
    future_states = np.zeros((len(pred_init), 1)).astype(int)
    future_trans = np.zeros((len(pred_init), 1)).astype(int)
    for idx in range(len(pred_init)):
        future_states[idx, 0] = pred_init[idx]
        future_trans[idx, 0] = trans_init[idx]
    # Iterate:
    step_count = 1
    while step_count <= max_iter - 1:
        future_states, future_trans = tree_build(future_states, future_trans, delta_p, Qp)
        step_count = step_count + 1
        q_val_check = np.zeros((future_states.shape[0],))
        # Keep only shortest paths so far:
        if recurs_cut == 1:
            # Pre-allocate space:
            future_states_recurs = np.empty((2, future_states.shape[1]), int)
            future_trans_recurs = np.empty((2, future_trans.shape[1]), int)
            # Obtain best q value (closest to initialization, q0, in LTL-based automaton):
            for idx in range(future_states.shape[0]):
                q_val_check[idx] = Qp[future_states[idx, -1], 2]
            q_best = min(q_val_check).astype(int)
            # Keep only the shortest-length satisfying trajectories:
            for idx in range(future_states.shape[0]):
                if Qp[future_states[idx, -1], 2] == q_best:
                    future_states_recurs = np.concatenate((future_states_recurs, [future_states[idx, :]]), axis=0)
                    future_trans_recurs = np.concatenate((future_trans_recurs, [future_trans[idx, :]]), axis=0)
            future_states = np.delete(future_states_recurs, [0, 1], 0)
            future_trans = np.delete(future_trans_recurs, [0, 1], 0)
    future_states_final = np.zeros((future_states.shape[0], future_states.shape[1] + 1))
    future_trans_final = np.zeros((future_trans.shape[0], future_trans.shape[1] + 1))
    for idx in range(future_states.shape[0]):
        future_states_final[idx, :] = np.append([final_prod_idx], future_states[idx, :])
        future_trans_final[idx, :] = np.append([0], future_trans[idx, :])
    # Compute final data:
    sol_prod_states = np.fliplr(future_states_final).astype(int)
    sol_trans = np.fliplr(future_trans_final).astype(int)
    # Compute the satisfying states in T:
    sol_T_states = which_T_state(Qp, sol_prod_states)
    return sol_prod_states, sol_T_states, sol_trans


def satisfying_paths_v2(initial_prod_idx, final_prod_idx, Qp, delta_p, recurs_cut):
    # Build the initial entries in array of states, + associated transitions that bring you from them to final state:
    trans_init, pred_init = unpack_delta_back(delta_p, Qp.shape[0], 4, final_prod_idx)
    future_states = np.zeros((len(pred_init), 1)).astype(int)
    future_trans = np.zeros((len(pred_init), 1)).astype(int)
    for idx in range(len(pred_init)):
        future_states[idx, 0] = pred_init[idx]
        future_trans[idx, 0] = trans_init[idx]
    # Iterate:
    stop_cond = 0
    max_iter = 30
    iter_count = 0
    while (stop_cond == 0) and (iter_count <= max_iter):
        iter_count = iter_count + 1
        future_states, future_trans = tree_build(future_states, future_trans, delta_p, Qp)
        q_val_check = np.zeros((future_states.shape[0],))
        Qp_val_check = np.zeros((future_states.shape[0],))
        # Keep only shortest paths so far:
        if recurs_cut == 1:
            # Pre-allocate space:
            future_states_recurs = np.empty((2, future_states.shape[1]), int)
            future_trans_recurs = np.empty((2, future_trans.shape[1]), int)
            # Obtain best q value (closest to initialization, q0, in LTL-based automaton):
            for idx in range(future_states.shape[0]):
                q_val_check[idx] = Qp[future_states[idx, -1], 2]
                Qp_val_check[idx] = future_states[idx, -1]
                if Qp_val_check[idx] == initial_prod_idx:
                    stop_cond = 1
            q_best = min(q_val_check).astype(int)
            # Keep only the shortest-length satisfying trajectories:
            for idx in range(future_states.shape[0]):
                if Qp[future_states[idx, -1], 2] == q_best:
                    future_states_recurs = np.concatenate((future_states_recurs, [future_states[idx, :]]), axis=0)
                    future_trans_recurs = np.concatenate((future_trans_recurs, [future_trans[idx, :]]), axis=0)
            future_states = np.delete(future_states_recurs, [0, 1], 0)
            future_trans = np.delete(future_trans_recurs, [0, 1], 0)
    if stop_cond == 0:
        future_states_final = np.zeros((future_states.shape[0], future_states.shape[1] + 1))
        future_trans_final = np.zeros((future_trans.shape[0], future_trans.shape[1] + 1))
        for idx in range(future_states.shape[0]):
            future_states_final[idx, :] = np.append([final_prod_idx], future_states[idx, :])
            future_trans_final[idx, :] = np.append([0], future_trans[idx, :])
        # Compute final data:
        sol_prod_states = np.fliplr(future_states_final).astype(int)
        sol_trans = np.fliplr(future_trans_final).astype(int)
        # Compute the satisfying states in T:
        sol_T_states = which_T_state(Qp, sol_prod_states)
    else:
        sat_idx_list = np.zeros((1,))
        for idx in range(future_states.shape[0]):
            if Qp[future_states[idx, -1], 0] == initial_prod_idx:
                sat_idx_list = np.append(sat_idx_list, [idx])
        sat_idx_list = np.delete(sat_idx_list, 0)
        future_states_final = np.zeros((len(sat_idx_list), future_states.shape[1] + 1))
        future_trans_final = np.zeros((len(sat_idx_list), future_trans.shape[1] + 1))
        for idx in range(len(sat_idx_list)):
            temp_idx = sat_idx_list[idx].astype(int)
            future_states_final[idx, :] = np.append([final_prod_idx], future_states[temp_idx, :])
            future_trans_final[idx, :] = np.append([0], future_trans[temp_idx, :])
        # Compute final data:
        sol_prod_states = np.fliplr(future_states_final).astype(int)
        sol_trans = np.fliplr(future_trans_final).astype(int)
        # Compute the satisfying states in T:
        sol_T_states = which_T_state(Qp, sol_prod_states)
    return sol_prod_states, sol_T_states, sol_trans


def main_test():
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

    # USER-SET TRANSITION SYSTEM:
    # See class definition "TransitionSystem"
    user_Obs = ['R', 'S1', 'S2', 'K1', 'D']  # Define all possible observations in T
    user_o = {'R': 10, 'S1': 3, 'S2': 0, 'K1': 6, 'D': [1, 2, 4, 5, 7, 8, 9, 11, 12]}  # Defining o(x) for all x in T
    print user_o[user_Obs[1]]
    # Notes:
    # - user_o indexed as user_o['D'] for example, or user_o[user_Obs[k]] for chosen index k
    # - The observation 'D' defines permissible but otherwise uninteresting regions in the domain
    delta_array = delta_trans(tri).astype(int)
    T = TransitionSystem(np.arange(0, tri_shape[0]), np.array([0, 1, 2, 3]), delta_array, user_Obs, user_o)
    print ''
    print 'TRANSITION SYSTEM DEFINED:'
    print 'X:'
    print T.X
    print 'Sigma:'
    print T.Sigma
    print 'delta:'
    print T.delta
    print 'O:'
    print T.Obs
    print 'o:'
    print T.o

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
    # Ordering: acdepting states have highest numbers, initial state is zero.
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
    print ''
    print 'LTL-BASED SYSTEM DEFINED:'
    print 'Q:'
    print A.Q
    print 'Q0:'
    print A.Q0
    print 'Sigma indices:'
    print A.Sigma
    print 'R (transition map for q, sigma):'
    print A.R
    print 'F:'
    print A.F

    # DEFINE PRODUCT AUTOMATON:
    # These lines immediately below need to be modified slightly for multi-valued A.Q0 and A.F:
    Qp = product_fcn(T.X, A.Q, len(Q)).astype(int)
    Q0p = product_fcn(T.X, [A.Q0], len(Q)).astype(int)
    Fp = product_fcn(T.X, [A.F], len(Q)).astype(int)
    print ''
    print 'PRODUCT AUTOMATON DEFINED: (Q_p, Q0_p, delta_p, F_p)'
    print 'Q_p: (#, x_idx, q_idx)'
    print Qp  # (Q_p index, x, q)
    print 'Q0_p:'
    print Q0p
    print 'F_p:'
    print Fp

    # Build product transition function:
    delta_p = delta_prod(Qp, T.delta, A.R, A.Q, user_o, alph_fcns).astype(int)
    print 'Transition function for product:'
    print delta_p

    # User-entered initial and final state (simplex ID #), and shortest path options:
    x0 = 2
    xf = 10
    initial_prod_idx = which_Qp_state(x0, Q0, Q)  # Here we are assuming there is only 1 q0 and 1 qf (simple LTL)
    final_prod_idx = which_Qp_state(xf, F, Q)
    shortest_paths = 1

    sol_prod_states, sol_T_states, sol_trans = satisfying_paths_v2(initial_prod_idx, final_prod_idx, Qp, delta_p, shortest_paths)
    # Display results:
    print ''
    print 'Satisfying state trajectory, in product automaton state IDs:'
    print sol_prod_states
    print ''
    print 'Satisfying order of states (simplices), in order:'
    print sol_T_states
    print ''
    print 'Satisfying control index "j" for u_j:'
    print sol_trans

    return

# Uncomment to test:
# main_test()
