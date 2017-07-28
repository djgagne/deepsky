import numpy as np
import pandas as pd
from deepsky.kriging import random_field_generator, spatial_covariance, distance_matrix, exp_kernel, gaussian_kernel
from deepsky.gan import generator_model, encoder_disc_model, train_linked_gan, stack_gen_disc, stack_enc_gen
from deepsky.gan import stack_gen_encoder, rescale_multivariate_data
import matplotlib.pyplot as plt
from datetime import datetime
import itertools as it
from os.path import join, exists
import os
from multiprocessing import Pool
import traceback
import keras.backend.tensorflow_backend as K
from keras.optimizers import Adam
from keras.models import Model


def main():
    gan_path = "/scratch/dgagne/random_gan_{0}/".format(datetime.utcnow().strftime("%Y%m%d"))
    gan_params = dict(generator_input_size=[16, 32, 128],
                      filter_width=[3, 5, 7],
                      min_data_width=[4],
                      min_conv_filters=[32, 64, 128],
                      batch_size=[256],
                      learning_rate=[0.0001],
                      activation=["relu", "selu", "leaky"],
                      dropout_alpha=[0, 0.05, 0.1],
                      beta_one=[0.5],
                      data_width=[32, 64],
                      train_size=[1024, 16384, 131072, 1048576],
                      length_scale=["full;3", "full;8", "full;3;8", "stacked;3;8;5", "combined;3;8",
                                     "blended;3;8"],
                      seed=[14268489],
                      )
    num_epochs = [1, 2, 3, 4, 5, 8, 10]
    num_gpus = 7
    out_dtype = "float32"
    metrics = ["accuracy", "binary_crossentropy"]
    gan_param_names = list(gan_params.keys())
    gan_param_combos = pd.DataFrame(list(it.product(*(gan_params[gan_name] for gan_name in gan_param_names))),
                                    columns=gan_param_names)
    gan_param_combos.to_csv(join(gan_path, "gan_param_combos.csv"), index_label="Index")
    pool = Pool(num_gpus)
    combo_ind = np.linspace(0, gan_param_combos.shape[0], num_gpus + 1).astype(int)
    if not exists(gan_path):
        os.mkdir(gan_path)
    for gpu_num in range(num_gpus):
        pool.apply_async(train_gan_config, (gpu_num,
                                            num_epochs,
                                            gan_param_combos.iloc[combo_ind[gpu_num]:combo_ind[gpu_num + 1]],
                                            metrics, gan_path, out_dtype))
    pool.close()
    pool.join()
    return


def train_gan_config(gpu_num, num_epochs, gan_params, metrics, gan_path, out_dtype):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(gpu_num)
        session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
                                                       gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                       log_device_placement=False))
        K.set_session(session)
        num_combos = gan_params.shape[0]
        with K.tf.device("/gpu:{0:d}".format(0)):
            for c, i in enumerate(gan_params.index.values):
                np.random.seed(gan_params.loc[i, "seed"])
                print("Starting combo {0:d} ({1:d} of {2:d})".format(i, c, num_combos))
                print(gan_params.loc[i])
                data = generate_random_fields(gan_params.loc[i, "train_size"],
                                              gan_params.loc[i, "data_width"],
                                              gan_params.loc[i, "length_scale"])
                scaled_data, scaling_values = rescale_multivariate_data(data)
                batch_size = int(gan_params.loc[i, "batch_size"])
                batch_diff = scaled_data.shape[0] % batch_size
                gen, vec_input = generator_model(input_size=int(gan_params.loc[i, "generator_input_size"]),
                                                 filter_width=int(gan_params.loc[i, "filter_width"]),
                                                 min_data_width=int(gan_params.loc[i, "min_data_width"]),
                                                 min_conv_filters=int(gan_params.loc[i, "min_conv_filters"]),
                                                 output_size=scaled_data.shape[1:],
                                                 stride=2,
                                                 activation=gan_params.loc[i, "activation"],
                                                 dropout_alpha=float(gan_params.loc[i, "dropout_alpha"]))
                disc, enc, image_input = encoder_disc_model(input_size=scaled_data.shape[1:],
                                                            filter_width=int(gan_params.loc[i, "filter_width"]),
                                                            min_data_width=int(gan_params.loc[i, "min_data_width"]),
                                                            min_conv_filters=int(gan_params.loc[i, "min_conv_filters"]),
                                                            output_size=int(gan_params.loc[i, "generator_input_size"]),
                                                            activation=gan_params.loc[i, "activation"],
                                                            dropout_alpha=float(gan_params.loc[i, "dropout_alpha"]))

                optimizer = Adam(lr=gan_params.loc[i, "learning_rate"],
                                 beta_1=gan_params.loc[i, "beta_one"])
                gen_model = Model(vec_input, gen)
                disc_model = Model(image_input, disc)
                enc_model = Model(image_input, enc)
                gen_model.compile(optimizer=optimizer, loss="mse")
                enc_model.compile(optimizer=optimizer, loss="mse")
                disc_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
                print("gen model")
                print(gen_model.summary())
                print("disc model")
                print(disc_model.summary())
                gen_disc_model = stack_gen_disc(gen_model, disc_model)
                gen_disc_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
                enc_gen_model = stack_enc_gen(enc_model, gen_model, disc_model)
                enc_gen_model.compile(optimizer=optimizer, loss="mse")
                print("gen model")
                print(gen_model.summary())
                print("disc model")
                print(disc_model.summary())
                print("gen disc model")
                print(gen_disc_model.summary())
                print("enc gen model")
                print(enc_gen_model.summary())
                history = train_linked_gan(scaled_data[:-batch_diff], gen_model, enc_model, disc_model,
                                           gen_disc_model, enc_gen_model,
                                           int(gan_params.loc[i, "generator_input_size"]),
                                           gan_path, i,
                                           batch_size=int(gan_params.loc[i, "batch_size"]),
                                           metrics=metrics,
                                           num_epochs=num_epochs, scaling_values=scaling_values,
                                           out_dtype=out_dtype)
                history.to_csv(join(gan_path, "gan_loss_history_{0:03d}.csv".format(i)), index_label="Time")

    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


def generate_random_fields(set_size, data_width, length_scale_str):
    length_scale_list = length_scale_str.split(";")
    spatial_pattern = length_scale_list[0]
    length_scales = [float(v) for v in length_scale_list[1:]]
    rand_gen = random_field_generator(data_width, data_width, length_scales, spatial_pattern=spatial_pattern)
    data = np.stack([next(rand_gen) for i in range(set_size)], axis=0)
    return data


def plot_random_fields():
    width = 32
    height = 32
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    test_distances = np.arange(1, 30)
    distances = distance_matrix(x_grid, y_grid)
    length_scales = np.array([2, 16])
    rand_gen = random_field_generator(x_grid, y_grid, length_scales, spatial_pattern="blended")
    rand_fields = [next(rand_gen) for x in range(25)]
    for rand_field in rand_fields:
        print(rand_field.std(), rand_field.mean())
    covariances = np.array([spatial_covariance(distances,
                                               rand_field,
                                               test_distances, 0.5) for rand_field in rand_fields])
    plt.figure(figsize=(6, 4))
    #plt.fill_between(test_distances, covariances.max(axis=0), covariances.min(axis=0), color='red', alpha=0.2)
    for cov_inst in covariances:
        plt.plot(test_distances, cov_inst, color='pink', marker='o', ls='-')
    plt.plot(test_distances, covariances.mean(axis=0), 'ro-')
    plt.plot(test_distances, exp_kernel(test_distances, length_scales[0]), 'bo-')
    fig, axes = plt.subplots(5, 5, figsize=(9, 9))
    plt.subplots_adjust(wspace=0,hspace=0)
    axef = axes.ravel()
    for a, ax in enumerate(axef):
        ax.contourf(rand_fields[a], np.linspace(-4, 4, 20), cmap="RdBu_r", extend="both")
        ax.tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off',
                       labelleft='off', labelbottom='off')
    plt.show()
    return


if __name__ == "__main__":
    main()