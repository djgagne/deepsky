import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, svdvals


def main():
    width = 64
    height = 64
    length_scales = np.array([4, 16, 32])
    z_in = np.random.normal(size=width * height)
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    out_fields = np.zeros((length_scales.size, height, width))
    for l, length_scale in enumerate(length_scales):
        out_fields[l] = random_field(x_grid, y_grid, z_in, length_scale).reshape(height, width)
    fig, axes = plt.subplots(1, length_scales.size, figsize=(15, 5))
    for a, ax in enumerate(axes):
        ax.contourf(x_grid, y_grid, out_fields[a], np.arange(-4, 4.1, 0.1), cmap="RdBu_r", extend="both")
        ax.set(title="Length Scale={0:d}".format(length_scales[a]))
    plt.show()
    #plt.contourf(x_grid, y_grid, out_fields.mean(axis=0), np.arange(-2, 2.5, 0.1), cmap="RdBu_r", extend="both")
    #plt.show()
    return


def random_field(x, y, z, length_scale):
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    distances = np.sqrt((x_flat - x_flat.T) ** 2 + (y_flat - y_flat.T) ** 2)
    corr = exp_kernel(distances, length_scale)
    cho_out = cholesky(corr, lower=True)
    svd_out = np.sqrt(svdvals(corr))
    print(svd_out.sum() / svd_out.max())
    out = np.dot(cho_out, z)
    return out


def exp_kernel(distance, length_scale):
    return np.exp(-distance / length_scale)

if __name__ == "__main__":
    main()