from deepsky.gan import wgan, generator_model, joint_discriminator_model, train_full_gan, encoder_model, rescale_multivariate_data
import numpy as np
import pandas as pd
from multiprocessing import Pool
import xarray as xr
from glob import glob
import itertools as it
import keras.backend.tensorflow_backend as K
from keras.optimizers import Adam
from keras.models import Model
from os.path import join, exists
import os
import traceback
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tsi", action="store_true", help="Load TSI data, otherwise load storm data")
    args = parser.parse_args()
    if args.tsi:
        data_path = "/scratch/dgagne/arm_tsi_sgp_nc/"
        variable_names = ["tsi_image"]
        gan_path = "/scratch/dgagne/arm_gan_{0}/".format(datetime.utcnow().strftime("%Y%m%d"))
        out_dtype = "uint8"
    else:
        data_path = "/scratch/dgagne/ncar_ens_storm_patches/"
        variable_names = ["composite_reflectivity_entire_atmosphere_prev",
                          "temperature_2_m_above_ground_prev",
                          "dew_point_temperature_2_m_above_ground_prev",
                          "u-component_of_wind_10_m_above_ground_prev",
                          "v-component_of_wind_10_m_above_ground_prev"]
        gan_path = "/scratch/dgagne/storm_gan_{0}/".format(datetime.utcnow().strftime("%Y%m%d"))
        out_dtype = "float32"
    gan_params = dict(generator_input_size=[8, 32, 128],
                      filter_width=[5],
                      min_data_width=[4],
                      min_conv_filters=[64, 128],
                      leaky_relu_alpha=[0.02],
                      batch_size=[256],
                      learning_rate=[0.0001],
                      beta_one=[0.2])
    num_epochs = [1, 2, 3]
    num_gpus = 6
    metrics = ("accuracy", "binary_crossentropy")
    total_combinations = 1
    if not exists(gan_path):
        os.mkdir(gan_path)
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
        pool.apply_async(evaluate_gan_config, (gpu_num, data_path, variable_names,
                                               num_epochs,
                                               gan_param_combos.iloc[combo_ind[gpu_num]:combo_ind[gpu_num + 1]],
                                               metrics, gan_path, out_dtype))
    pool.close()
    pool.join()
    return


def evaluate_gan_config(gpu_num, data_path, variable_names, num_epochs, gan_params, metrics, gan_path, out_dtype):
    """
    
    
    Args:
        gpu_num: 
        data_path: 
        variable_names: 
        num_epochs: 
        gan_params: 
        metrics: 
        gan_path: 
        out_dtype: 

    Returns:

    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(gpu_num)
        print("Loading data {0}".format(gpu_num))
        if "tsi" in data_path:
            data = load_tsi_data(data_path, variable_names)
        else:
            data = load_storm_patch_data(data_path, variable_names)
        max_vals = data.max(axis=0).max(axis=0).max(axis=0)
        min_vals = data.min(axis=0).min(axis=0).min(axis=0)
        print("Rescaling data {0}".format(gpu_num))
        scaled_data = rescale_multivariate_data(data)
        session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
                                                               gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                               log_device_placement=False))
        K.set_session(session)
        with K.tf.device("/gpu:{0:d}".format(0)):
            for i in gan_params.index.values:
                print("Starting combo {0:d}".format(i))
                print(gan_params.loc[i])
                batch_size = int(gan_params.loc[i, "batch_size"])
                batch_diff = scaled_data.shape[0] % batch_size
                gen, vec_input = generator_model(input_size=int(gan_params.loc[i, "generator_input_size"]),
                                                 filter_width=int(gan_params.loc[i, "filter_width"]),
                                                 min_data_width=int(gan_params.loc[i, "min_data_width"]),
                                                 min_conv_filters=int(gan_params.loc[i, "min_conv_filters"]),
                                                 output_size=scaled_data.shape[1:],
                                                 stride=2)
                enc, image_input = encoder_model(input_size=scaled_data.shape[1:],
                                                 filter_width=int(gan_params.loc[i, "filter_width"]),
                                                 min_data_width=int(gan_params.loc[i, "min_data_width"]),
                                                 min_conv_filters=int(gan_params.loc[i, "min_conv_filters"]),
                                                 output_size=int(gan_params.loc[i, "generator_input_size"]))
                combined, disc = joint_discriminator_model(gen,
                                                           enc,
                                                           image_input,
                                                           vec_input,
                                                           stride=2,
                                                           filter_width=int(gan_params.loc[i, "filter_width"]),
                                                           min_conv_filters=int(gan_params.loc[i, "min_conv_filters"]),
                                                           min_data_width=int(gan_params.loc[i, "min_data_width"]),
                                                           leaky_relu_alpha=gan_params.loc[i, "leaky_relu_alpha"])
                optimizer = Adam(lr=gan_params.loc[i, "learning_rate"],
                                beta_1=gan_params.loc[i, "beta_one"])
                gen_model = Model(vec_input, gen)
                enc_model = Model(image_input, enc)
                gen_model.compile(optimizer=optimizer, loss="mse")
                enc_model.compile(optimizer=optimizer, loss="mse")
                disc.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=list(metrics))
                for layer in combined.layers:
                    if layer in disc.layers:
                        layer.trainable = False
                combined.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=list(metrics))
                print(disc.summary())
                print(combined.summary())
                train_full_gan(scaled_data[:-batch_diff], gen_model, enc_model, disc, combined,
                                         int(gan_params.loc[i, "generator_input_size"]),
                                         gan_path, i,
                                         batch_size=int(gan_params.loc[i, "batch_size"]),
                                         metrics=metrics,
                                         num_epochs=num_epochs, max_vals=max_vals,
                                         min_vals=min_vals, out_dtype=out_dtype)
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


def load_tsi_data(data_path, variable_names, width=32, r_patch=(100, 100, 150, 150),
                  c_patch=(280, 120, 280, 120)):
    data_patches = []
    data_files = sorted(glob(join(data_path, "*.nc")))
    variable_name = variable_names[0]
    for data_file in data_files:
        ds = xr.open_dataset(data_file)
        for i in range(len(r_patch)):
            data_patches.append(ds[variable_name][:,
                                                  r_patch[i]:r_patch[i] + width,
                                                  c_patch[i]:c_patch[i] + width].values)
        ds.close()
    data = np.vstack(data_patches)
    return data


def load_storm_patch_data(data_path, variable_names):
    data_patches = []
    data_files = sorted(glob(join(data_path, "*.nc")))
    for data_file in data_files:
        print(data_file)
        ds = xr.open_dataset(data_file)
        patch_arr = []
        for variable in variable_names:
            patch_arr.append(ds[variable].values)
        data_patches.append(np.stack(patch_arr, axis=-1))
    data = np.vstack(data_patches)
    return data


if __name__ == "__main__":
    main()
