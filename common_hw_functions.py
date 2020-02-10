# INITIAL IMPORTING, PATHS, SETTINGS
import math
import numpy as np


def mean_diff_slope(local_points):
    # For a cluster of 5 ordered points, determine median differential slope sd at inner point
    # data_shape = local_points.shape  # should be 5x2
    # if data_shape[0] != 5:
    #     local_points = np.transpose(local_points)
    # Compute outer slope:
    s_star = (local_points[4, 1] - local_points[0, 1])/(local_points[4, 0] - local_points[0, 0])
    s_inner = (local_points[3, 1] - local_points[1, 1])/(local_points[3, 0] - local_points[1, 0])
    # Compute median differential slope:
    sd = s_inner - s_star
    return sd


def mean_diff_slope_v2(local_points):
    # For a cluster of 4 ordered points, determine median differential slope sd at average inner point
    # data_shape = local_points.shape  # should be 4x2
    # if data_shape[0] != 4:
    #     local_points = np.transpose(local_points)
    print(local_points)
    # Compute outer slope:
    s_star = (local_points[3, 1] - local_points[0, 1])/(local_points[3, 0] - local_points[0, 0])
    s_inner = (local_points[2, 1] - local_points[1, 1])/(local_points[2, 0] - local_points[1, 0])
    # Compute median differential slope:
    sd = s_inner - s_star
    return sd


def build_sd_array(data_array):
    # For any number of points, return the array of ordered sd values
    # data_shape = data_array.shape  # should be Nx2
    # if data_shape[1] != 2:
    #     data_array = np.transpose(data_array)
    # Compute sd array:
    sd_array = np.empty((len(data_array[:, 0]) - 4, 2))
    for idx2 in range(len(sd_array[:, 0])):
        idx1 = idx2 + 2
        sd_array[idx2, 0] = data_array[idx1, 0]  # x value of center point
        sd_array[idx2, 1] = mean_diff_slope(data_array[idx1 - 2: idx1 + 3, :])  # select 5 points
    return sd_array


def build_sd_array_v2(data_array):
    # For any number of points, return the array of ordered sd values
    # data_shape = data_array.shape  # should be Nx2
    # if data_shape[1] != 2:
    #     data_array = np.transpose(data_array)
    # Compute sd array:
    sd_array = np.empty((len(data_array[:, 0]) - 4, 2))
    for idx2 in range(len(sd_array[:, 0])):
        idx1 = idx2 + 2
        sd_array[idx2, 0] = np.mean(data_array[idx1 - 2: idx1 + 2, 0])  # x value of center point
        sd_array[idx2, 1] = mean_diff_slope_v2(data_array[idx1 - 2: idx1 + 2, :])  # select 4 points
    return sd_array


def cluster_average_data(data_array, n_pts):
    # Note this will discard remainder of data (n < n_pts) from end, if data doesn't divide evenly
    avg_array = np.empty((int(math.floor(len(data_array[:, 0])/n_pts)), 2))
    for idx2 in range(len(avg_array[:, 0])):
        idx1 = n_pts*idx2
        avg_array[idx2, :] = np.mean(data_array[idx1: idx1 + n_pts, :], axis=0)
    return avg_array


def rms_erb(data_vec):
    # data_vec = np.abs(data_vec)
    # # Custom RMS function, not sure if one already exists in a package
    # ssq = reduce(lambda i, j: i + j * j, [data_vec[:1][0] ** 2.] + data_vec[1:])
    # rms = np.sqrt(ssq/len(data_vec))
    rms = np.sqrt(np.mean(data_vec ** 2.))
    return rms


def dx(f, x):
    return abs(0.0 - f(x))


def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - (f(x0) / df(x0))
        delta = dx(f, x0)
    return x0


def saveFigurePDF(figureName, plt, path):
    #   Borrowed from BSK
    figFileName = path+figureName+".pdf"
    # if not os.path.exists(os.path.dirname(figFileName)):
    #     try:
    #         os.makedirs(os.path.dirname(figFileName))
    #     except OSError as exc:  # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    plt.savefig(figFileName, transparent=True, bbox_inches='tight', dpi=300, pad_inches=0.05)


def main_test():
    # Short code to test functions in this file as needed
    # Build test array:
    test_array = np.array([[0.0, -209.573], [250.0, -210.073], [500.0, -210.073], [750.0, -211.073], [1000.0, -211.073],
                           [1250.0, -209.573], [1500.0, -209.573]])
    sd_array_test = build_sd_array(test_array)
    print(sd_array_test)
    print(int(math.floor(12/3)))
    avg_array_test = cluster_average_data(test_array, 3)
    print(avg_array_test)
    test_vec = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Testing one-line sum of squares w/ lambda function:
    ssq = np.sqrt(np.mean(test_vec ** 2.))
    print(ssq)
    return


# Execute function testing:
# main_test()