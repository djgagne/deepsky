import numpy as np
import pandas as pd
from netCDF4 import Dataset
from multiprocessing import Pool
from os.path import exists, join
import argparse
from deepsky.evaluation import spatial_covariance, local_spatial_covariance
from deepsky.kriging import exp_kernel, distance_matrix, random_field_generator
from scipy.linalg import cholesky
from scipy.stats import ttest_ind_from_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to random GAN files")
    parser.add_argument("-o", "--out", help="Output path for evaluation data")
    parser.add_argument("-e", "--epochs", default="1,2,3,4,5,8,10", help="comma separated list of epochs")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    gan_params = pd.read_csv(join(args.input, "gan_param_combos.csv"), index_col="Index")
    epochs = np.array(args.epochs.split(","), dtype=int)
    eval_distances = np.arange(1, 9)
    tolerance = 0.2
    gan_cov_stats = []
    if args.proc > 1:
        pool = Pool(args.proc)
        for gan_index in gan_params.index:
            pool.apply_async(eval_full_gan_config, (gan_index, epochs, args.input,
                                                    gan_params.loc[gan_index, "data_width"],
                                                    eval_distances,
                                                    tolerance),
                             callback=gan_cov_stats.append)
        pool.close()
        pool.join()
    else:
        for gan_index in gan_params.index:
            gan_cov_stats.append(eval_full_gan_config(gan_index, epochs, args.input,
                                                      gan_params.loc[gan_index, "data_width"],
                                                      eval_distances,
                                                      tolerance))
    all_gan_cov_stats = pd.concat(gan_cov_stats)
    all_gan_cov_stats.to_csv(join(args.out, "gan_cov_stats.csv"), index_label="Index")
    return


def eval_full_gan_config(gan_index, epochs, gan_path, data_width, eval_distances, tolerance):
    gan_mean_cols = ["cov_mean_{0:02d}".format(x) for x in eval_distances]
    gan_sd_cols = ["cov_sd_{0:02d}".format(x) for x in eval_distances]
    gan_t_cols = ["cov_t_{0:02d}".format(x) for x in eval_distances]
    gan_stat_cols = ["Index", "Epoch"] + gan_mean_cols + gan_sd_cols + gan_t_cols + ["mean_tscore", "max_tscore"]
    gan_stats = pd.DataFrame(np.ones((len(epochs), len(gan_stat_cols))) * np.nan,
                             columns=gan_stat_cols, index=np.arange(len(epochs)))
    x = np.arange(data_width)
    y = np.arange(data_width)
    x_g, y_g = np.meshgrid(x, y)
    distances = distance_matrix(x_g, y_g)
    corr = exp_kernel(distances, 3)
    cho = cholesky(corr, lower=True)
    cho_inv = np.linalg.inv(cho)
    rand_covs = np.zeros((256, eval_distances.size))
    random_gen = random_field_generator(x_g, y_g, [3])
    random_fields = np.stack([next(random_gen) for x in range(256)], axis=0)[:, :, :, 0]
    random_noise = np.stack([np.matmul(cho_inv,
                                       random_field.reshape(random_field.size, 1)).reshape(random_fields.shape[1:]) for
                             random_field in random_fields], axis=0)
    noise_cov = np.zeros((256, eval_distances.size))
    for p, patch in enumerate(random_noise):
        noise_cov[p] = spatial_covariance(distances, patch, eval_distances, tolerance)
    noise_mean = noise_cov.mean(axis=0)
    noise_sd = noise_cov.std(axis=0)
    for e, epoch in enumerate(epochs):
        gan_patch_file = join(gan_path, "gan_gen_patches_{0:04d}_epoch_{1:04d}.nc".format(gan_index, epoch))
        if exists(gan_patch_file):
            gan_patch_ds = Dataset(gan_patch_file)
            gan_patches = np.array(gan_patch_ds.variables["gen_patch"][:, :, :, 0])
            gan_patch_ds.close()
            for p, patch in enumerate(gan_patches):
                rand_patch = np.matmul(cho_inv, patch.reshape(patch.size, 1)).reshape(patch.shape)
                rand_covs[p] = spatial_covariance(distances, rand_patch, eval_distances, tolerance)
            gan_stats.loc[e, gan_mean_cols] = rand_covs.mean(axis=0)
            gan_stats.loc[e, gan_sd_cols] = rand_covs.std(axis=0)
            gan_stats.loc[e, gan_t_cols] = ttest_ind_from_stats(gan_stats.loc[e, gan_mean_cols].values,
                                                                    gan_stats.loc[e, gan_sd_cols].values,
                                                                    np.array([256] * len(eval_distances)),
                                                                    noise_mean,
                                                                    noise_sd,
                                                                    np.array([256] * len(eval_distances)),
                                                                    equal_var=False)[0]
            gan_stats.loc[e, "mean_tscore"] = np.abs(gan_stats.loc[e, gan_t_cols]).mean()
            gan_stats.loc[e, "max_tscore"] = np.abs(gan_stats.loc[e, gan_t_cols]).max()
    return gan_stats

if __name__ == "__main__":
    main()