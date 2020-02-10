# domain_functions.py
#
# Use:
#   For Hybrid Systems final project. Set of functions to generate/process domain.
#   Also developed control-to-facet steering law
#
# Requirements:
#   Packages: numpy, scipy, itertools, matplotlib
#   Codes: N/A

# INITIAL IMPORTING, PATHS
import numpy as np
import scipy
from scipy import spatial
from scipy import optimize
from numpy import ndarray
import matplotlib.pyplot as plt
import itertools


def cost_fun_v1(x, c):
    u1 = x[0:2]
    u2 = x[2:4]
    u3 = x[4:6]
    J = (c - np.linalg.norm(u1))**2. + (c - np.linalg.norm(u2))**2. + (c - np.linalg.norm(u3))**2.
    # J = (c - np.linalg.norm(u1) - np.linalg.norm(u2) - np.linalg.norm(u3))**2.
    return J


def nhat_tri_facet(tri, num):
    # Returns normal vectors for facets emanating CCW from a vertex ("generating vertex")
    # Normal vectors listed (3 x 2) by row in same order as generating vertex listed in vertices_here (def. below)
    # To do: permute output --> normal vector numbering matches opposite vertex numbering
    points = tri.points
    vertices = tri.simplices
    vertices_here = vertices[num, :]
    R_r = np.array([[0., 1.], [-1., 0.]])  # Turn right 90 degrees
    nhat_array = np.zeros((3, 2))
    for idx in range(0, 3):
        if idx != 2:
            v_temp = points[vertices_here[idx+1], :] - points[vertices_here[idx], :]  # CCW
        else:
            v_temp = points[vertices_here[0], :] - points[vertices_here[idx], :]
        vhat_temp = v_temp/np.linalg.norm(v_temp)
        nhat_temp = np.matmul(R_r, np.transpose(vhat_temp))
        nhat_array[idx, :] = nhat_temp
        permute = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])  # See notes
    return np.matmul(permute, nhat_array)


def cons_array_for_F(exit_facet_idx, nhat_array, B):
    # Develop the set of linear constraints to determine control to a facet specified by exit_facet_idx
    if exit_facet_idx == 0:
        C_row1 = np.concatenate((-np.matmul(np.transpose(nhat_array[1, :]), B), np.zeros((4,))), axis=0)
        C_row2 = np.concatenate((-np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((4,))), axis=0)
        C_row3 = C_row1 + C_row2
        C_row4 = np.concatenate((np.zeros((2,)), np.matmul(np.transpose(nhat_array[0, :]), B), np.zeros((2,))), axis=0)
        C_row5 = np.concatenate((np.zeros((2,)), -np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((2,))), axis=0)
        C_row6 = np.concatenate((np.zeros((4,)), np.matmul(np.transpose(nhat_array[0, :]), B)), axis=0)
        C_row7 = np.concatenate((np.zeros((4,)), -np.matmul(np.transpose(nhat_array[1, :]), B)), axis=0)
    elif exit_facet_idx == 1:
        C_row1 = np.concatenate((np.matmul(np.transpose(nhat_array[1, :]), B), np.zeros((4,))), axis=0)
        C_row2 = np.concatenate((-np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((4,))), axis=0)
        C_row3 = np.concatenate((np.zeros((2,)), -np.matmul(np.transpose(nhat_array[0, :]), B), np.zeros((2,))), axis=0)
        C_row4 = np.concatenate((np.zeros((2,)), -np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((2,))), axis=0)
        C_row5 = C_row3 + C_row4
        C_row6 = np.concatenate((np.zeros((4,)), np.matmul(np.transpose(nhat_array[1, :]), B)), axis=0)
        C_row7 = np.concatenate((np.zeros((4,)), -np.matmul(np.transpose(nhat_array[0, :]), B)), axis=0)
    else: # Finish this
        C_row1 = np.concatenate((np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((4,))), axis=0)
        C_row2 = np.concatenate((-np.matmul(np.transpose(nhat_array[1, :]), B), np.zeros((4,))), axis=0)
        C_row3 = np.concatenate((np.zeros((2,)), np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((2,))), axis=0)
        C_row4 = np.concatenate((np.zeros((2,)), -np.matmul(np.transpose(nhat_array[0, :]), B), np.zeros((2,))), axis=0)
        C_row5 = np.concatenate((np.zeros((4,)), -np.matmul(np.transpose(nhat_array[0, :]), B)), axis=0)
        C_row6 = np.concatenate((np.zeros((4,)), -np.matmul(np.transpose(nhat_array[1, :]), B)), axis=0)
        C_row7 = C_row5 + C_row6
    C_cons = np.array([C_row1, C_row2, C_row3, C_row4, C_row5, C_row6, C_row7])
    return C_cons


def cons_array_for_stay(nhat_array, B):
    C_row1 = np.concatenate((np.zeros((2,)), -np.matmul(np.transpose(nhat_array[0, :]), B), np.zeros((2,))), axis=0)
    C_row2 = np.concatenate((np.zeros((4,)), -np.matmul(np.transpose(nhat_array[0, :]), B)), axis=0)
    C_row3 = np.concatenate((-np.matmul(np.transpose(nhat_array[1, :]), B), np.zeros((4,))), axis=0)
    C_row4 = np.concatenate((np.zeros((4,)), -np.matmul(np.transpose(nhat_array[1, :]), B)), axis=0)
    C_row5 = np.concatenate((-np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((4,))), axis=0)
    C_row6 = np.concatenate((np.zeros((2,)), -np.matmul(np.transpose(nhat_array[2, :]), B), np.zeros((2,))), axis=0)
    C_cons = np.array([C_row1, C_row2, C_row3, C_row4, C_row5, C_row6])
    return C_cons


def u_for_F_v1(tri, num, c, B, facet_id):
    cost_fun = lambda x: cost_fun_v1(x, c)
    x0 = np.zeros((6,))
    nhat_array = nhat_tri_facet(tri, num)
    # Extract vertex coordinates:
    points = tri.points
    vertices = tri.simplices
    vertices_here = vertices[num, :]
    C_cons = cons_array_for_F(facet_id, nhat_array, B)
    m = 7  # Number of constraints for control to facet with no dynamics
    cons = optimize.LinearConstraint(C_cons, np.zeros(m), np.inf*np.ones(m))
    res = optimize.minimize(cost_fun, x0, method='trust-constr', constraints=cons)
    local_debug = 0
    if local_debug == 1:
        print ''
        print 'Results of optimization (success, cost, number of iterations):'
        print res.success
        print res.fun
        print res.nit
    Vrow1 = np.concatenate((points[vertices_here[0], :], np.ones((1,))), axis=0)
    Vrow2 = np.concatenate((points[vertices_here[1], :], np.ones((1,))), axis=0)
    Vrow3 = np.concatenate((points[vertices_here[2], :], np.ones((1,))), axis=0)
    Vmat = np.array([Vrow1, Vrow2, Vrow3])
    u_optimal = res.x
    RHS = np.array([u_optimal[0:2], u_optimal[2:4], u_optimal[4:6]])
    f_and_g_trans = np.matmul(np.linalg.inv(Vmat), RHS)
    f_and_g = np.transpose(f_and_g_trans)
    # print 'f_and_g:'
    # print f_and_g
    return u_optimal, f_and_g


def u_for_stay_v1(tri, num, c, B):
    cost_fun = lambda x: cost_fun_v1(x, c)
    x0 = np.zeros((6,))
    nhat_array = nhat_tri_facet(tri, num)
    # Extract vertex coordinates:
    points = tri.points
    vertices = tri.simplices
    vertices_here = vertices[num, :]
    C_cons = cons_array_for_stay(nhat_array, B)
    m = 6  # Number of constraints for control to facet with no dynamics
    cons = optimize.LinearConstraint(C_cons, np.zeros(m), np.inf*np.ones(m))
    res = optimize.minimize(cost_fun, x0, method='trust-constr', constraints=cons)
    local_debug = 0
    if local_debug == 1:
        print ''
        print 'Results of optimization (success, cost, number of iterations):'
        print res.success
        print res.fun
        print res.nit
    Vrow1 = np.concatenate((points[vertices_here[0], :], np.ones((1,))), axis=0)
    Vrow2 = np.concatenate((points[vertices_here[1], :], np.ones((1,))), axis=0)
    Vrow3 = np.concatenate((points[vertices_here[2], :], np.ones((1,))), axis=0)
    Vmat = np.array([Vrow1, Vrow2, Vrow3])
    u_optimal = res.x
    RHS = np.array([u_optimal[0:2], u_optimal[2:4], u_optimal[4:6]])
    f_and_g_trans = np.matmul(np.linalg.inv(Vmat), RHS)
    f_and_g = np.transpose(f_and_g_trans)
    return u_optimal, f_and_g


def fg_for_domain(tri, c_val, B, opt):
    # Option 1: control to facet
    # Option 2: "stay in simplex"
    n_tri = tri.points.size/3
    if opt == 1:
        control_array = np.zeros((2*n_tri, 9))
        for idx1 in range(0, n_tri):
            for idx2 in range(0, 3):
                u_optimal_temp, fg_array_temp = u_for_F_v1(tri, idx1, c_val, B, idx2)
                control_array[2*idx1:2*(idx1+1), 3*idx2:3*(idx2+1)] = fg_array_temp
    else:
        control_array = np.zeros((2*n_tri, 3))
        for idx1 in range(0, n_tri):
            u_optimal_temp, fg_array_temp = u_for_stay_v1(tri, idx1, c_val, B)
            control_array[2*idx1:2*(idx1+1), :] = fg_array_temp
    return control_array


def test_4d():
    # Break the 4D unit cube into simplices...
    iterables = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
    nodes = []
    for t in itertools.product(*iterables):
        nodes.append(np.array(t))
    tri = spatial.Delaunay(np.array(nodes))
    print ''
    print tri.points  # Numbering of the points is in order of input
    print ''
    print tri.simplices  # From point numbers, give the triangles, starting with # 0
    print ''
    print tri.neighbors  # Give the triangle number of neighbors, and -1 for boundary
    return


def main_test():
    # As a main function, this function tests other functions written in domain_functions.py:
    # - compute triangulation of domain
    # -
    nodes = np.array([[0., 0.], [0., 10.], [15., 10.], [15., 0.], [6., 10.], [9., 10.], [11., 0.], [2., 8.],
                      [8., 7.], [11., 2.], [5., 5.]])
    tri = spatial.Delaunay(nodes)
    print ''
    print 'Entered points, triangles by vertex, and neighbors:'
    print tri.points  # Numbering of the points is in order of input
    print ''
    print tri.simplices  # From point numbers, give the triangles, starting with # 0
    print ''
    print tri.neighbors  # Give the triangle number of neighbors, and -1 for boundary
    # Test which triangle:
    # print tri.find_simplex(np.array([2., 2.]))

    # Choose a simplex and facet control idx:
    num = 0
    facet_id = 0

    # Extract points:
    points = tri.points
    vertices = tri.simplices
    vertices_here = vertices[num, :]
    print ''
    print 'vertices_here:'
    print vertices_here

    # Extract normals:
    nhat_array = nhat_tri_facet(tri, num)
    print ''
    print 'Local normal vectors:'
    print nhat_array

    u_optimal, fg_array = u_for_F_v1(tri, num, 0.01, np.eye(2), facet_id)
    u_stay, fg_array_stay = u_for_stay_v1(tri, num, 0.01, np.eye(2))
    print ''
    print 'Optimal u at each vertex:'
    print u_optimal
    print ''
    print 'fg_array for select triangle and facet:'
    print fg_array
    print ''
    print 'Optimal u to stay:'
    print u_stay

    fg_master_array = fg_for_domain(tri, 0.01, np.eye(2), 2)
    print ''
    print 'fg_master_array:'
    print fg_master_array

    x_array = points[vertices_here, 0]
    y_array = points[vertices_here, 1]
    x_lim = np.array([ndarray.min(x_array), ndarray.max(x_array)])
    y_lim = np.array([ndarray.min(y_array), ndarray.max(y_array)])

    domain_plot = 1
    control_plot = 1
    # Plot partitioning of domain:
    if domain_plot == 1:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.triplot(nodes[:, 0], nodes[:, 1], tri.simplices.copy())
        plt.plot(nodes[:, 0], nodes[:, 1], 'o')
        plt.title('Partitioned Domain')
        plt.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    # Quiver velocity field to leave selected triangle:
    if control_plot == 1:
        X, Y = np.meshgrid(np.arange(x_lim[0]-0.5, x_lim[1]+0.5, 0.5), np.arange(y_lim[0]-0.5, y_lim[1]+0.5, 0.5))
        U = fg_array[0, 0]*X + fg_array[0, 1]*Y + fg_array[0, 2]
        V = fg_array[1, 0]*X + fg_array[1, 1]*Y + fg_array[1, 2]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title('Control Vector Field for Chosen Simplex - Exit Chosen Facet')
        ax.triplot(nodes[:, 0], nodes[:, 1], tri.simplices.copy())
        ax.plot(nodes[:, 0], nodes[:, 1], 'o')
        ax.quiver(X, Y, U, V)
        ax.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
        # ax.set_xlim(x_lim)
        # ax.set_ylim(y_lim)

        X, Y = np.meshgrid(np.arange(x_lim[0]-0.5, x_lim[1]+0.5, 0.5), np.arange(y_lim[0]-0.5, y_lim[1]+0.5, 0.5))
        U = fg_array_stay[0, 0]*X + fg_array_stay[0, 1]*Y + fg_array_stay[0, 2]
        V = fg_array_stay[1, 0]*X + fg_array_stay[1, 1]*Y + fg_array_stay[1, 2]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title('Control Vector Field for Chosen Simplex - Remain')
        ax.triplot(nodes[:, 0], nodes[:, 1], tri.simplices.copy())
        ax.plot(nodes[:, 0], nodes[:, 1], 'o')
        ax.quiver(X, Y, U, V)
        ax.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)
        # ax.set_xlim(x_lim)
        # ax.set_ylim(y_lim)
    if (domain_plot == 1) or (control_plot == 1):
        plt.show()
    return


# Execute test:
# main_test()