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


def tscore(cov_a, cov_b):
    return ttest_ind(cov_a, cov_b)