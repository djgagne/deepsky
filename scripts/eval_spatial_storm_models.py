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
from multiprocessing import Pool


def main():

    return


def evaluate_single_cnn_model(config, data_path, input_variables, output_variable, out_path):
    storm_data, storm_centers, storm_dates = load_storm_patch_data(data_path, input_variables)
    unique_dates = storm_dates.unique()

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


def hail_cnn(data_width=32, num_input_channels=1, filter_width=5, min_conv_filters=16,
             filter_growth_rate=2, min_data_width=4,
             dropout_alpha=0, activation="relu", regularization_alpha=0.01):
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
    return Model(cnn_input, cnn_model)




if __name__ == "__main__":
    main()