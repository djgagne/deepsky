from deepsky.gan import generator_model, discriminator_model, train_gan, rescale_data, encoder_model
import numpy as np
import pandas as pd
from multiprocessing import Pool
import xarray as xr
import itertools as it
import keras.backend as K
from keras.optimizers import Adam
from os.path import join


def main():
    data_path = "/scratch/dgagne/arm_tsi_sgp_nc/"
    variable_name = "tsi_image"
    gan_path = "/scratch/dgagne/arm_gan/"
    gan_params = dict(generator_input_size=[10, 100],
                      filter_width=[5],
                      min_data_width=[2, 4],
                      max_conv_filters=[256, 512, 1024],
                      min_conv_filters=[8, 16, 32],
                      leaky_relu_alpha=[0, 0.1, 0.2],
                      batch_size=[128],
                      learning_rate=[0.001, 0.0001, 0.00001],
                      beta_one=[0.5, 0.9])
    num_epochs = [10, 100, 500, 1000]
    num_gpus = 8
    total_combinations = 1
    for param_name, values in gan_params.items():
        total_combinations *= len(values)
    print(total_combinations)
    gan_param_names = list(gan_params.keys())
    gan_param_combos = pd.DataFrame(list(it.product(*(gan_params[gan_name] for gan_name in gan_param_names))),
                                    columns=gan_param_names)
    gan_param_combos.to_csv(join(gan_path, "gan_param_combos.csv"), index_label="Index")
    pool = Pool(num_gpus)
    combo_ind = np.linspace(0, gan_param_combos.shape[0], num_gpus + 1).astype(int)
    for gpu_num in range(num_gpus):
        pool.apply_async(evaluate_gan_config, (gpu_num, data_path, variable_name,
                                               num_epochs,
                                               gan_param_combos.loc[combo_ind[gpu_num]:combo_ind[gpu_num + 1]],
                                               gan_path))
    pool.close()
    pool.join()
    return


def evaluate_gan_config(gpu_num, data_path, variable_name, num_epochs, gan_params, gan_path):
    data = load_tsi_data(data_path, variable_name)
    scaled_data = rescale_data(data)
    with K.tf.device("/gpu:{0:d}".format(gpu_num)):
        for i in gan_params.index:
            gen = generator_model(batch_size=gan_params.loc[i, "batch_size"],
                                  input_size=gan_params.loc[i, "generator_input_size"],
                                  filter_width=gan_params.loc[i, "filter_width"],
                                  min_data_width=gan_params.loc[i, "min_data_width"],
                                  max_conv_filters=gan_params.loc[i, "max_conv_filters"],
                                  output_size=scaled_data.shape[1:],
                                  stride=2)
            disc = discriminator_model(input_size=scaled_data.shape[1:],
                                       stride=2,
                                       filter_width=gan_params.loc[i, "filter_width"],
                                       min_conv_filters=gan_params.loc[i, "min_conv_filters"],
                                       min_data_width=gan_params.loc[i, "min_data_width"],
                                       leaky_relu_alpha=gan_params.loc[i, "leaky_relu_alpha"])
            enc = encoder_model(input_size=scaled_data.shape[1:],
                                filter_width=gan_params.loc[i, "filter_width"],
                                min_data_width=gan_params.loc[i, "min_data_width"],
                                max_conv_filters=gan_params.loc[i, "max_conv_filters"],
                                output_size=gan_params.loc[i, "generator_input_size"])
            optimizer = Adam(lr=gan_params.loc[i, "learning_rate"],
                             beta_1=gan_params.loc[i, "beta_one"])
            history = train_gan(scaled_data, gen, disc, gan_path, i,
                                batch_size=gan_params.loc[i, "batch_size"],
                                num_epochs=num_epochs,
                                gen_optimizer=optimizer, disc_optimizer=optimizer,
                                encoder=enc)
            history.to_csv(join(gan_path, "gan_loss_history_{0:06d}.csv".format(i)), index_label="Step")
    return


def load_tsi_data(data_path, variable_name, width=32, r_patch=(100, 100, 150, 150),
                  c_patch=(280, 120, 280, 120)):
    ds = xr.open_mfdataset(data_path + "*.nc")
    data_patches = []
    for i in range(len(r_patch)):
        data_patches.append(ds[variable_name][:,
                                              r_patch[i]:r_patch[i] + width,
                                              c_patch[i]:c_patch[i] + width].values)
    ds.close()
    data = np.vstack(data_patches)
    return data

if __name__ == "__main__":
    main()