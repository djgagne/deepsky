import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Input, Conv1D, Merge, concatenate
from keras.layers import Activation, Reshape, LeakyReLU, concatenate, Dropout, BatchNormalization, AlphaDropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.models import load_model
import keras.backend as K
from os.path import exists, join
import os
from deepsky.gan import train_linked_gan, stack_gen_disc, stack_gen_enc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def normalize_multivariate_data(data, scaling_values=None):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: 4-dimensional array with dimensions (example, y, x, channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
    for i in range(data.shape[-1]):
        scaling_values.loc[i, ["mean", "std"]] = [data[:, :, :, i].mean(), data[:, :, :, i].std()]
        normed_data[:, :, :, i] = (data[:, :, :, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
    return normed_data, scaling_values


def unnormalize_multivariate_data(normed_data, scaling_values):
    """
    Return normalized data to original values by multiplying by standard deviation and adding the mean.

    Args:
        normed_data: 4-dimensional array of normalized values with dimensions (example, y, x, channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        data array
    """
    data = np.zeros(normed_data.shape, dtype=normed_data.dtype)
    for i in range(normed_data.shape[-1]):
        data[:, :, :, i] = normed_data[:, :, :, i] * scaling_values.loc[i, "std"] + scaling_values.loc[i, "mean"]
    return data

def load_storm_patch_data(data_path, variable_names):
    data_patches = []
    centers = []
    valid_dates = []
    data_files = sorted(glob(join(data_path, "*.nc")))
    for data_file in data_files:
        print(data_file)
        ds = xr.open_dataset(data_file)
        patch_arr = []
        all_vars = list(ds.variables.keys())
        if np.all(np.in1d(variable_names, all_vars)):
            centers.append(np.array([ds["longitude"][:, 32, 32], ds["latitude"][:, 32, 32]]).T)
            valid_dates.append(ds["valid_date"].values)
            for variable in variable_names:
                patch_arr.append(ds[variable][:, 16:-16, 16:-16].values)
            data_patches.append(np.stack(patch_arr, axis=-1))
        ds.close()
        del patch_arr
        del ds
    center_arr = np.vstack(centers)
    valid_date_index = pd.DatetimeIndex(np.concatenate(valid_dates))
    data = np.vstack(data_patches)
    return data, center_arr, valid_date_index

def generator_model(input_size=100, filter_width=5, min_data_width=4,
                    min_conv_filters=64, output_size=(32, 32, 1), stride=2, activation="relu",
                    output_activation="linear", dropout_alpha=0):
    """ 
    Creates a generator convolutional neural network for a generative adversarial network set. The keyword arguments
    allow aspects of the structure of the generator to be tuned for optimal performance.

    Args:
        input_size (int): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the first convolved layer after the input layer
        min_conv_filters (int): Number of convolutional filters in the last convolutional layer
        output_size (tuple of size 3): Dimensions of the output
        stride (int): Number of pixels that the convolution filter shifts between operations.
        activation (str): Type of activation used for convolutional layers. Use "leaky" for Leaky ReLU.
        output_activation (str): Type of activation used on the output layer
        dropout_alpha (float): proportion of nodes dropped out
    Returns:
        Model output graph, model input
    """
    num_layers = int(np.log2(output_size[0]) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers - 1)) 
    curr_conv_filters = max_conv_filters
    vector_input = Input(shape=(input_size, ), name="gen_input")
    model = Dense(units=max_conv_filters * min_data_width * min_data_width, kernel_regularizer=l2())(vector_input)
    model = Reshape((min_data_width, min_data_width, max_conv_filters))(model)
    if activation == "leaky":
        model = LeakyReLU(alpha=0.2)(model)
    else:
        model = Activation(activation)(model)
    for i in range(1, num_layers):
        curr_conv_filters //= 2
        model = Conv2DTranspose(curr_conv_filters, filter_width,
                                  strides=(stride, stride), padding="same")(model)
        if activation == "leaky":
            model = LeakyReLU(alpha=0.2)(model)
        else:
            model = Activation(activation)(model)
        if activation == "selu":
            model = AlphaDropout(dropout_alpha)(model)
        else:
            model = Dropout(dropout_alpha)(model)
    model = Conv2DTranspose(output_size[-1], filter_width,
                              strides=(stride, stride),
                              padding="same")(model)
    model = Activation(output_activation)(model)
    return model, vector_input

def encoder_disc_model(input_size=(32, 32, 1), filter_width=5, min_data_width=4,
                       min_conv_filters=64, output_size=100, stride=2, activation="relu",
                       encoder_output_activation="linear",
                       dropout_alpha=0):
    """
    Creates an encoder/discriminator convolutional neural network that reproduces the generator input vector.
    The keyword arguments allow aspects of the structure of the enocder/discriminator to be tuned
    for optimal performance.

    Args:
        input_size (tuple of ints): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the last convolved layer
        min_conv_filters (int): Number of convolutional filters in the first convolutional layer
        output_size (int): Dimensions of the output
        stride (int): Number of pixels that the convolution filter shifts between operations.
        activation (str): Type of activation used for convolutional layers. Use "leaky" for Leaky ReLU.
        encoder_output_activation (str): Type of activation used on the output layer
        dropout_alpha (float): Proportion of nodes dropped out during training.
    Returns:
        discriminator model output, encoder model output, image input
    """
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    image_input = Input(shape=input_size, name="enc_input")
    model = image_input
    for c in range(num_layers):
        model = Conv2D(curr_conv_filters, filter_width,
                       strides=(stride, stride), padding="same")(model)
        if activation == "leaky":
            model = LeakyReLU(0.2)(model)
        else:
            model = Activation(activation)(model)
        if activation == "selu":
            model = AlphaDropout(dropout_alpha)(model)
        else:
            model = Dropout(dropout_alpha)(model)
        curr_conv_filters *= 2
    model = Flatten()(model)
    enc_model = Dense(256, kernel_regularizer=l2())(model)
    if activation == "leaky":
        enc_model = LeakyReLU(0.2)(enc_model)
    else:
        enc_model = Activation(activation)(enc_model)
    enc_model = Dense(output_size, kernel_regularizer=l2())(enc_model)
    enc_model = Activation(encoder_output_activation)(enc_model)
    disc_model = Dense(1, kernel_regularizer=l2())(model)
    disc_model = Activation("sigmoid")(disc_model)
    return disc_model, enc_model, image_input

def main():
    ua_vars = ['geopotential_height_500_mb_prev',
           'geopotential_height_700_mb_prev',
           'geopotential_height_850_mb_prev',
           'temperature_500_mb_prev',
             'temperature_700_mb_prev',
             'temperature_850_mb_prev',
             'dew_point_temperature_500_mb_prev',
            'dew_point_temperature_700_mb_prev',
             'dew_point_temperature_850_mb_prev',
             'u-component_of_wind_500_mb_prev',
             'u-component_of_wind_700_mb_prev',
             'u-component_of_wind_850_mb_prev',
             'v-component_of_wind_500_mb_prev',
             'v-component_of_wind_700_mb_prev',
             'v-component_of_wind_850_mb_prev']
    ua_data, ua_centers, ua_dates = load_storm_patch_data("/scratch/dgagne/ncar_ens_storm_patches/", ua_vars)
    ua_norm, ua_scaling = normalize_multivariate_data(ua_data)
    train_indices = np.load("/scratch/dgagne/storm_ua_gan/train_indices.npy")
    test_indices = np.load("/scratch/dgagne/storm_ua_gan/test_indices.npy")
    batch_size = 32
    batch_diff = train_indices.shape[0] % batch_size
    session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
                                                            gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                            log_device_placement=False))
    K.set_session(session)
    with K.tf.device("/gpu:{0:d}".format(0)):
        metrics = ["accuracy"]
        num_epochs = [1, 5, 10]
        gen, vec_input = generator_model(input_size=64, min_conv_filters=32, min_data_width=4,
                                        filter_width=3, output_size=(32, 32, 15), activation="selu")
        disc, enc, image_input = encoder_disc_model(input_size=(32, 32, 15), min_conv_filters=32, output_size=64, 
                                                    min_data_width=4, filter_width=3,
                                                    activation="selu")
        optimizer = Adam(lr=0.0001, beta_1=0.5)
        gen_model = Model(vec_input, gen)
        disc_model = Model(image_input, disc)
        enc_model = Model(image_input, enc)
        gen_model.compile(optimizer=optimizer, loss="mse")
        enc_model.compile(optimizer=optimizer, loss="mse")
        disc_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
        gen_disc_model = stack_gen_disc(gen_model, disc_model)
        gen_disc_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
        gen_enc_model = stack_gen_enc(gen_model, enc_model, disc_model)
        gen_enc_model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])
        print("gen model")
        print(gen_model.summary())
        print("disc model")
        print(disc_model.summary())
        print("gen disc model")
        print(gen_disc_model.summary())
        print("enc gen model")
        print(gen_enc_model.summary())

        gan_path = "/scratch/dgagne/storm_ua_gan/"
        train_linked_gan(ua_norm[train_indices[:train_indices.shape[0] - batch_diff]], gen_model, enc_model, disc_model,
                        gen_disc_model, gen_enc_model,
                        64,
                        gan_path, 0, batch_size=batch_size,
                        metrics=metrics, num_epochs=num_epochs, scaling_values=ua_scaling,
                        out_dtype=np.float32, ind_encoder=None)
    session.close()

if __name__ == "__main__":
    main()
