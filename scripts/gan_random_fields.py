import numpy as np
import pandas as pd
from deepsky.kriging import random_field_generator, spatial_covariance, distance_matrix, exp_kernel
from deepsky.gan import generator_model, discriminator_model, train_gan_quiet, stack_gen_disc, stack_gen_enc
from deepsky.gan import normalize_multivariate_data, encoder_model
import matplotlib.pyplot as plt
import itertools as it
from os.path import join, exists
import os
from multiprocessing import Pool
import traceback
import keras.backend.tensorflow_backend as K
from keras.optimizers import Adam


def main():
    
    #gan_path = "/glade/scratch/dgagne/random_gan/"
    gan_path = "/orangefs/scratch/dgagne/random_gan/"
    if not exists(gan_path):
        os.mkdir(gan_path)
    path_files = os.listdir(gan_path)
    if len(path_files) > 0:
        for path_file in path_files:
            os.remove(join(gan_path, path_file))
    #gan_path = "/scratch/dgagne/random_gan_{0}/".format(datetime.utcnow().strftime("%Y%m%d"))
    #gan_path = "/scratch/dgagne/random_gan_{0}".format("20170905")
    # gan_params = dict(generator_input_size=[16, 32, 128],
    #                   filter_width=[3, 5],
    #                   min_data_width=[4],
    #                   min_conv_filters=[32, 64, 128],
    #                   batch_size=[256],
    #                   learning_rate=[0.0001],
    #                   activation=["relu", "selu", "leaky"],
    #                   dropout_alpha=[0, 0.05, 0.1],
    #                   beta_one=[0.5],
    #                   data_width=[32],
    #                   train_size=[16384, 131072],
    #                   length_scale=["full;3", "full;8", "full;3;8", "stacked;3;8;5", "combined;3;8",
    #                                  "blended;3;8"],
    #                   seed=[14268489],
    #                   )
    gan_params = dict(generator_input_size=[32, 128],
                      filter_width=[5],
                      min_data_width=[4],
                      min_conv_filters=[32, 64],
                      batch_size=[256],
                      learning_rate=[0.001, 0.0001],
                      activation=["relu", "leaky"],
                      dropout_alpha=[0, 0.1, 0.5],
                      use_dropout=[True],
                      use_noise=[False],
                      beta_one=[0.5],
                      data_width=[32],
                      train_size=[131072],
                      #train_size=[4096],
                      length_scale=["full;3", "full;8",
                                    "blended;3;8"],
                      stride=[1, 2],
                      seed=[14268489],
                      )
    num_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_gpus = 4
    out_dtype = "float32"
    metrics = ["accuracy", "binary_crossentropy"]
    if not exists(gan_path):
        os.mkdir(gan_path)
    gan_param_names = sorted(list(gan_params.keys()))
    gan_param_combos = pd.DataFrame(list(it.product(*(gan_params[gan_name] for gan_name in gan_param_names))),
                                    columns=gan_param_names)
    gan_param_combos.to_csv(join(gan_path, "gan_param_combos.csv"), index_label="Index")
    pool = Pool(num_gpus)
    combo_ind = np.linspace(0, gan_param_combos.shape[0], num_gpus + 1).astype(int)
    for gpu_num in range(num_gpus):
        pool.apply_async(train_gan_configs, (gpu_num,
                                             num_epochs,
                                             gan_param_combos.iloc[combo_ind[gpu_num]:combo_ind[gpu_num + 1]],
                                             metrics, gan_path, out_dtype))
    pool.close()
    pool.join()
    return


def train_gan_configs(gpu_num, num_epochs, gan_params, metrics, gan_path, out_dtype):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(gpu_num)
        for c, i in enumerate(gan_params.index.values):
            session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
                                                           gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                           log_device_placement=False))
            K.set_session(session)
            num_combos = gan_params.shape[0]
            with K.tf.device("/gpu:{0:d}".format(0)):
                if not exists(join(gan_path, "gan_gen_patches_{0:04d}_epoch_{1:04d}.nc".format(i, num_epochs[-1]))):
                    print("Starting combo {0:d} ({1:d} of {2:d})".format(i, c, num_combos))
                    train_single_gan(i, num_epochs, gan_params, metrics, gan_path)
                else:
                    print("Already trained combo {0:d} ({1:d} of {2:d})".format(i, c, num_combos))
            session.close()
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


def train_single_gan(gan_index, num_epochs, gan_params, metrics, gan_path):
    print(gan_params.loc[gan_index])
    np.random.seed(gan_params.loc[gan_index, "seed"])
    data, scaling_values = normalize_multivariate_data(generate_random_fields(gan_params.loc[gan_index, "train_size"],
                                                                              gan_params.loc[gan_index, "data_width"],
                                                                              gan_params.loc[gan_index, "length_scale"]
                                                                              ))
    scaling_values.to_csv(join(gan_path, "scaling_values_{0:04d}.csv".format(gan_index)), index_label="Channel")
    batch_size = int(gan_params.loc[gan_index, "batch_size"])
    batch_diff = data.shape[0] % batch_size
    if batch_diff > 0:
        data = data[:data.shape[0]-batch_diff]
    print("create gan models")
    gen_model = generator_model(input_size=int(gan_params.loc[gan_index, "generator_input_size"]),
                                filter_width=int(gan_params.loc[gan_index, "filter_width"]),
                                min_data_width=int(gan_params.loc[gan_index, "min_data_width"]),
                                min_conv_filters=int(gan_params.loc[gan_index, "min_conv_filters"]),
                                output_size=data.shape[1:],
                                stride=int(gan_params.loc[gan_index, "stride"]),
                                activation=gan_params.loc[gan_index, "activation"],
                                dropout_alpha=float(gan_params.loc[gan_index, "dropout_alpha"]))
    disc_model = discriminator_model(input_size=data.shape[1:],
                                     filter_width=int(gan_params.loc[gan_index, "filter_width"]),
                                     min_data_width=int(gan_params.loc[gan_index, "min_data_width"]),
                                     min_conv_filters=int(gan_params.loc[gan_index, "min_conv_filters"]),
                                     activation=gan_params.loc[gan_index, "activation"],
                                     stride=int(gan_params.loc[gan_index, "stride"]),
                                     dropout_alpha=float(gan_params.loc[gan_index, "dropout_alpha"]))
    ind_enc_model = encoder_model(input_size=data.shape[1:],
                                  filter_width=int(gan_params.loc[gan_index, "filter_width"]),
                                  min_data_width=int(gan_params.loc[gan_index, "min_data_width"]),
                                  min_conv_filters=int(gan_params.loc[gan_index, "min_conv_filters"]),
                                  output_size=int(gan_params.loc[gan_index, "generator_input_size"]),
                                  activation=gan_params.loc[gan_index, "activation"],
                                  stride=int(gan_params.loc[gan_index, "stride"]),
                                  dropout_alpha=float(gan_params.loc[gan_index, "dropout_alpha"]))
    optimizer = Adam(lr=gan_params.loc[gan_index, "learning_rate"],
                     beta_1=gan_params.loc[gan_index, "beta_one"])
    gen_model.compile(optimizer=optimizer, loss="mse")
    ind_enc_model.compile(optimizer=optimizer, loss="mse")
    disc_model.compile(optimizer=optimizer, loss="binary_crossentropy")
    ind_enc_model.compile(optimizer=optimizer, loss="mse")
    gen_disc_model = stack_gen_disc(gen_model, disc_model)
    gen_disc_model.compile(optimizer=optimizer, loss="binary_crossentropy")
    gen_enc_model = stack_gen_enc(gen_model, ind_enc_model)
    gen_enc_model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])
    print("gen model")
    print(gen_model.summary())
    print("disc model")
    print(disc_model.summary())
    print("gen disc model")
    print(gen_disc_model.summary())
    print("enc gen model")
    print(gen_enc_model.summary())
    history = train_gan_quiet(data, gen_model, disc_model,
                              ind_enc_model, gen_disc_model, gen_enc_model,
                              int(gan_params.loc[gan_index, "generator_input_size"]),
                              int(gan_params.loc[gan_index, "batch_size"]),
                              num_epochs, gan_index, gan_path)
    history.to_csv(join(gan_path, "gan_history_{0:04d}.csv".format(gan_index)), index_label="Index")
    del data


def generate_random_fields(set_size, data_width, length_scale_str):
    length_scale_list = length_scale_str.split(";")
    spatial_pattern = length_scale_list[0]
    length_scales = [float(v) for v in length_scale_list[1:]]
    x = np.arange(data_width)
    y = np.arange(data_width)
    x_grid, y_grid = np.meshgrid(x, y)
    rand_gen = random_field_generator(x_grid, y_grid, length_scales, spatial_pattern=spatial_pattern)
    print("Generate stack of random fields")
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
    # plt.fill_between(test_distances, covariances.max(axis=0), covariances.min(axis=0), color='red', alpha=0.2)
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
