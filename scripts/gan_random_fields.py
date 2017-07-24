import numpy as np
from deepsky.kriging import random_field_generator, spatial_covariance, distance_matrix, exp_kernel
import matplotlib.pyplot as plt


def main():
    width = 32
    height = 32
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    distances = distance_matrix(x_grid, y_grid)
    length_scales = np.array([8])
    rand_gen = random_field_generator(x_grid, y_grid, length_scales)
    rand_fields = [next(rand_gen) for x in range(9)]
    test_distances = np.arange(1, 30)
    covariances = np.array([spatial_covariance(distances,
                                               rand_field,
                                               test_distances, 0.1) for rand_field in rand_fields])
    plt.figure(figsize=(6, 4))
    plt.plot(test_distances, covariances.mean(axis=0), 'ko-')
    plt.plot(test_distances, exp_kernel(test_distances, length_scales[0]), 'bo-')
    plt.show()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axef = axes.ravel()
    for a, ax in enumerate(axef):
        ax.contourf(rand_fields[a], np.linspace(-4, 4, 20), cmap="RdBu_r")
    plt.show()
    return


if __name__ == "__main__":
    main()