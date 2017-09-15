import numpy as np
from numba import jit
from scipy.stats import ttest_ind


@jit(nopython=True)
def spatial_covariance(distances, z, eval_distances, tolerance=0.2):
    """
    Calculate the empirical covariances among all points that are a certain distance apart.

    Args:
        distances: Square distance matrix between all points in terms of number of grid points
        z: Intensity values at each point
        eval_distances: Distance values at which covariance is calculated
        tolerance:

    Returns:
        Spatial covariance values for each eval_distance value.
    """
    if distances[np.triu_indices(distances.shape[0])].max() > 1000:
        sub_distances = distances
    else:
        sub_distances = np.array(distances, copy=True)
        sub_distances[np.triu_indices(sub_distances.shape[0])] = 999999
    covariances = np.zeros(eval_distances.size)
    z_flat = z.ravel()
    for d, eval_distance in enumerate(eval_distances):
        points_a, points_b = np.where(np.abs(sub_distances - eval_distance) <= tolerance)
        covariances[d] = np.sum((z_flat[points_a] - z_flat[points_a].mean()) *
                                (z_flat[points_b] - z_flat[points_b].mean())) / (float(points_a.size) - 1.0)
        covariances[d] /= z_flat[points_a].std() * z_flat[points_b].std()
    return covariances


@jit(nopython=True)
def local_spatial_covariance(window_width, stride, distances, z, eval_distances, tolerance=0.2):
    """
    Calculate spatial covariance values within a moving window over a spatial domain.

    Args:
        window_width: width of the spatial window in number of grid points
        stride: how far to advance the window between covariance calculations
        distances: Pointwise distance matrix
        z: Intensity values being evaluated. Should be in 2D grid shape
        eval_distances: Set of distances being evaluated
        tolerance: Bounds for capturing points within distance +/- tolerance value.

    Returns:
        Grid of covariance values with dimensions (eval_distances,
                                                   z.shape[0] - window_width + 1) // stride,
                                                   z.shape[1] - window_width + 1) // stride)
    """
    num_windows_col = (z.shape[1] - window_width + 1) // stride
    num_windows_row = (z.shape[0] - window_width + 1) // stride
    cov_grid = np.zeros((len(eval_distances), num_windows_row, num_windows_col))
    w_i = 0
    w_j = 0
    c_i = 0
    c_j = 0
    index_grid = np.arange(z.size).reshape(z.shape)
    while w_i < z.shape[0] - window_width:
        while w_j < z.shape[1] - window_width:
            d_points = index_grid[w_i: w_i + window_width, w_j: w_j + window_width].ravel()
            cov_grid[:, c_i, c_j] = spatial_covariance(distances[d_points, d_points],
                                                       z[w_i: w_i + window_width, w_j: w_j + window_width],
                                                       eval_distances, tolerance=tolerance)
            w_j += stride
            c_j += 1
        w_i += stride
        c_j += 1
    return cov_grid

