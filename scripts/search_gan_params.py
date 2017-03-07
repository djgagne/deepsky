from deepsky.gan import generator_model, discriminator_model, stack_gen_disc, train_gan, rescale_data
import numpy as np
import pandas as pd
from multiprocessing import Pool
import xarray as xr
import itertools as it


def main():
    gan_params = dict(generator_input_size=[10, 100],
                      filter_width=[5],
                      min_data_width=[2, 4],
                      max_conv_filters=[256, 512, 1024],
                      min_conv_filters=[8, 16, 32],
                      leaky_relu_alpha=[0, 0.1, 0.2],
                      learning_rate=[0.001, 0.0002],
                      beta_one=[0.5, 0.9])
    num_epochs = [10, 100, 1000]
    num_gpus = 8
    total_combinations = 1
    for param_name, values in gan_params.items():
        total_combinations *= len(values)
    print(total_combinations)
    pool = Pool(num_gpus)
    gan_param_names = list(gan_params.keys())
    gan_param_combos = it.product(*(gan_params[gan_name] for gan_name in gan_param_names))
    for combo in gan_param_combos:
        print(combo)

    return


def evaluate_gan_config(data_path, variable_name, gan_path, num_epochs, gan_param_combos):
    data = load_data(data_path, variable_name)
    scaled_data = rescale_data(data)
    return


def load_data(data_path, variable_name):
    ds = xr.open_mfdataset(data_path + "*.nc")
    data = ds[variable_name].values
    ds.close()
    return data

if __name__ == "__main__":
    main()