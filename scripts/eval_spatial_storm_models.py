import numpy as np
import pandas as pd
import xarray as xr
from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Input, Conv1D, Merge, concatenate
from keras.layers import Activation, Reshape, LeakyReLU, concatenate, Dropout, BatchNormalization, AlphaDropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
import keras.backend as K
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from deepsky.gan import normalize_multivariate_data, unnormalize_multivariate_data
from multiprocessing import Pool
from glob import glob
from os.path import join, exists
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as config_obj:
        config = yaml.load(config_obj)
    pool = Pool(args.proc)

    return


def train_split_generator(values, train_split, num_samples):
    split_index = int(np.round(train_split * values.size))
    for n in range(num_samples):
        shuffled_values = np.random.permutation(values)
        train_values = shuffled_values[:split_index]
        test_values = shuffled_values[split_index:]
        yield train_values, test_values


def evaluate_conv_net(data_path, input_variables, output_variable, output_mask,
                      out_path, sampling_config, conv_net_config):
    storm_data, storm_centers, storm_dates = load_storm_patch_data(data_path, input_variables)
    storm_norm_data, storm_scaling_values = normalize_multivariate_data(storm_data)
    output_data, output_centers, output_dates = load_storm_patch_data(data_path, [output_variable, output_mask])
    max_hail = np.array([output_data[i, :, :, 0][output_data[i, :, :, 1] > 0].max()
                         for i in range(output_data.shape[0])])
    hail_labels = np.zeros(max_hail.shape)
    unique_dates = storm_dates.unique()
    storm_sampler = train_split_generator(unique_dates, sampling_config["train_split"],
                                          sampling_config["num_samples"])
    for n in range(sampling_config["num_samples"]):
        train_dates, test_dates = next(storm_sampler)
        train_indices = np.where(np.in1d(storm_dates, train_dates))[0]
        test_indices = np.where(np.in1d(storm_dates, test_indices))[0]
        hail_conv_net_model = hail_conv_net(**conv_net_config)
        hail_conv_net_model.fit(storm_norm_data[train_indices], hail_labels,
                                batch_size=conv_net_config["batch_size"],
                                epochs=conv_net_config["num_epochs"], verbose=2)
        hail_conv_net_model.predict(storm_norm_data[test_indices])
    return


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


def hail_conv_net(data_width=32, num_input_channels=1, filter_width=5, min_conv_filters=16,
                  filter_growth_rate=2, min_data_width=4,
                  dropout_alpha=0, activation="relu", regularization_alpha=0.01, optimizer="sgd",
                  learning_rate=0.001, loss="mse", metrics=("mae", "auc"), **kwargs):
    cnn_input = Input(shape=(data_width, data_width, num_input_channels))
    num_conv_layers = int(np.log2(data_width) - np.log2(min_data_width))
    num_filters = min_conv_filters
    cnn_model = cnn_input
    for c in range(num_conv_layers):
        cnn_model = Conv2D(num_filters, filter_width, strides=2, padding="same",
                           kernel_regularizer=l2(regularization_alpha))(cnn_model)
        if activation == "leaky":
            cnn_model = LeakyReLU(0.2)(cnn_model)
        else:
            cnn_model = Activation(activation)(cnn_model)
        cnn_model = BatchNormalization()(cnn_model)
        cnn_model = Dropout(dropout_alpha)(cnn_model)
        num_filters = int(num_filters * filter_growth_rate)
    cnn_model = Flatten()(cnn_model)
    cnn_model = Dense(1)(cnn_model)
    cnn_model = Activation("sigmoid")(cnn_model)
    cnn_model_complete = Model(cnn_input, cnn_model)
    if optimizer.lower() == "sgd":
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        opt = Adam(lr=learning_rate, beta_1=0.5)
    metrics = list(metrics)
    if "auc" in metrics:
        metrics[metrics.index("auc")] = K.tf.metrics.auc
    cnn_model_complete.compile(optimizer=opt, loss=loss, metrics=metrics)
    return cnn_model_complete



if __name__ == "__main__":
    main()