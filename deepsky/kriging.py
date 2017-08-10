import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, svdvals
from numba import jit


def main():
    width = 32
    height = 32
    length_scales = np.array([4, 16, 32])
    z_in = np.random.normal(size=width * height)
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    out_fields = np.zeros((length_scales.size + 1, height, width))
    for l, length_scale in enumerate(length_scales):
        out_fields[l] = random_field(x_grid, y_grid, z_in, length_scale).reshape(height, width)
    out_fields[-1] = out_fields[:-1].mean(axis=0)
    fig, axes = plt.subplots(1, length_scales.size + 1, figsize=(20, 5))
    for a, ax in enumerate(axes):
        ax.contourf(x_grid, y_grid, out_fields[a], np.arange(-4, 4.1, 0.1), cmap="RdBu_r", extend="both")
        if a >= len(length_scales):
            ax.set(title="Combined")
        else:
            ax.set(title="Length Scale={0:d}".format(length_scales[a]))
    plt.show()
    #plt.contourf(x_grid, y_grid, out_fields.mean(axis=0), np.arange(-2, 2.5, 0.1), cmap="RdBu_r", extend="both")
    #plt.show()
    return


def exp_kernel(distance, length_scale):
    """
    Exponential spatial covariance function

    Args:
        distance: array of distances among points
        length_scale: Scaling factor for exponential covariance. Larger values result in stronger covariances in space.

    Returns:
        Covariance value for each distance
    """
    return np.exp(-distance / length_scale)


def gaussian_kernel(distance, length_scale):
    return np.exp(-(distance / length_scale) ** 2)


def distance_matrix(x, y):
    """
    Calculate distances among points.

    Args:
        x: array of x-coordinate values
        y: array of y-coordinate values

    Returns:
        Square matrix of distances among points
    """
    x_flat = x.reshape(-1, 1).astype(np.float64)
    y_flat = y.reshape(-1, 1).astype(np.float64)
    return np.sqrt((x_flat - x_flat.T) ** 2 + (y_flat - y_flat.T) ** 2)


def random_field(x, y, z, length_scale):
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    distances = np.sqrt((x_flat - x_flat.T) ** 2 + (y_flat - y_flat.T) ** 2)
    corr = exp_kernel(distances, length_scale)
    cho_out = cholesky(corr, lower=True)
    svd_out = np.sqrt(svdvals(corr))
    print(svd_out.sum() / svd_out.max())
    out = np.dot(cho_out, z.ravel())
    return out


def random_field_generator(x, y, length_scales, spatial_pattern="full"):
    """
    Calculates cholesky decomposition of given coordinates and then generates fields from Gaussian random noise
    with specified covariance structures. Four different spatial patterns are available:
    * full: each field has the same covariance across all points. The covariance is randomly chosen from the
            provided options.
    * stacked: Each channel has a different specified covariance.
    * combined: Fields with different covariances are averaged together
    * blended: Fields with different covariances are blended with a logistic function.

    Args:
        x: all x-coordinates for every point
        y:  all y-coordinates for every point
        length_scales: Length scales of covariances
        spatial_pattern: How covariances of different length scales are combined.

    Returns:
        A generated random field each time the generator is called.
    """
    distances = distance_matrix(x, y)
    correlations = []
    cho_matrices = []
    for ls in length_scales:
        correlations.append(exp_kernel(distances, ls))
        cho_matrices.append(cholesky(correlations[-1], lower=True))
    while True:
        if spatial_pattern == "full":
            scale = np.random.randint(0, len(length_scales))
            yield np.dot(cho_matrices[scale], np.random.normal(size=(x.size, 1))).reshape(x.shape[0],
                                                                                          x.shape[1], 1)
        elif spatial_pattern == "stacked":
            yield np.stack([np.dot(cho_matrices[scale],
                                   np.random.normal(size=x.size)).reshape(x.shape[0],
                                                                          x.shape[1])
                            for scale in range(len(length_scales))], axis=-1)
        elif spatial_pattern == "combined":
            yield np.stack([np.dot(cho_matrices[scale],
                            np.random.normal(size=x.size)).reshape(x.shape[0], x.shape[1])
                            for scale in range(len(length_scales))], axis=-1).mean(axis=-1).reshape(x.shape[0],
                                                                                                    x.shape[1], 1)
        elif spatial_pattern == "blended":
            fields = np.stack([np.dot(cho_matrices[scale],
                               np.random.normal(size=x.size)).reshape(x.shape[0], x.shape[1])
                               for scale in range(len(length_scales))], axis=0)
            rows, cols = np.indices(fields[0].shape)
            logistic_cols = 1.0 / (1 + np.exp(-2 * (cols - cols.mean())))
            yield np.reshape(logistic_cols * fields[0] + (1-logistic_cols) * fields[1], (x.shape[0], x.shape[1], 1))


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

if __name__ == "__main__":
    main()
