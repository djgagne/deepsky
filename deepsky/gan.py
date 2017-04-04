from __future__ import division
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, BatchNormalization
from keras.layers import Activation, Reshape, LeakyReLU
import xarray as xr
from os.path import join


def generator_model(input_size=100, filter_width=5, min_data_width=4,
                    max_conv_filters=256, output_size=(32, 32, 1), stride=2):
    """
    Creates a generator convolutional neural network for a generative adversarial network set. The keyword arguments
    allow aspects of the structure of the generator to be tuned for optimal performance.

    Args:
        input_size (int): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the first convolved layer after the input layer
        max_conv_filters (int): Number of convolutional filters in the first convolutional layer
        output_size (tuple of size 3): Dimensions of the output
        stride (int): Number of pixels that the convolution filter shifts between operations.

    Returns:
        Keras convolutional neural network.
    """
    data_widths = [min_data_width]
    conv_filters = [max_conv_filters]
    while data_widths[-1] < output_size[0] // stride:
        data_widths.append(data_widths[-1] * stride)
        conv_filters.append(conv_filters[-1] // stride)
    model = Sequential()
    model.add(Dense(input_shape=(input_size,), output_dim=max_conv_filters * min_data_width * min_data_width))
    model.add(Reshape((min_data_width, min_data_width, max_conv_filters)))
    for i in range(1, len(data_widths)):
        model.add(Conv2DTranspose(conv_filters[i], filter_width,
                                  strides=(stride, stride), padding="same"))
        model.add(Activation("relu"))
    model.add(Conv2DTranspose(output_size[-1], filter_width,
                              strides=(stride, stride),
                              padding="same"))
    model.add(Activation("tanh"))
    return model


def encoder_model(input_size=(32, 32, 1), filter_width=5, min_data_width=4,
                    max_conv_filters=256, output_size=100, stride=2):
    """
    Creates an encoder convolutional neural network that reproduces the generator input vector. The keyword arguments
    allow aspects of the structure of the generator to be tuned for optimal performance.

    Args:
        input_size (tuple of ints): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the first convolved layer after the input layer
        max_conv_filters (int): Number of convolutional filters in the first convolutional layer
        output_size (int): Dimensions of the output
        stride (int): Number of pixels that the convolution filter shifts between operations.

    Returns:
        Keras convolutional neural network.
    """
    data_widths = [min_data_width]
    conv_filters = [max_conv_filters]
    while data_widths[-1] <= input_size[0] // stride:
        data_widths.append(data_widths[-1] * stride)
        conv_filters.append(conv_filters[-1] // stride)
    model = Sequential()
    for i in range(len(data_widths)-1, 0, -1):
        if i == len(data_widths) - 1:
            model.add(Conv2D(conv_filters[i], filter_width,
                                    input_shape=input_size,
                                    strides=(stride, stride), padding="same"))
        else:
            model.add(Conv2D(conv_filters[i], filter_width,
                                    strides=(stride, stride), padding="same"))
        model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(output_size))
    model.add(Activation("tanh"))
    return model


def discriminator_model(input_size=(32, 32, 1), stride=2, filter_width=5,
                        max_conv_filters=16, min_data_width=4, leaky_relu_alpha=0.2):
    """
    Creates a discriminator model for a generative adversarial network.

    Args:
        input_size (tuple of size 3): Dimensions of input data
        stride (int): Number of pixels the convolution filter is shifted between operations
        filter_width (int): Width of convolution filters
        max_conv_filters (int): Number of convolution filters in the last layer. Halves in each previous layer
        min_data_width (int): Smallest width of input data after convolution downsampling before flattening
        leaky_relu_alpha (float): scaling coefficient for negative values in Leaky Rectified Linear Unit

    Returns:
        Keras generator model
    """
    data_widths = [min_data_width]
    conv_filters = [max_conv_filters]
    while data_widths[-1] <= input_size[0] // stride:
        data_widths.append(data_widths[-1] * stride)
        conv_filters.append(conv_filters[-1] // stride)
    conv_filters = conv_filters[::-1]
    model = Sequential()
    for c, conv_count in enumerate(conv_filters):
        if c == 0:
            model.add(Conv2D(conv_count, filter_width, input_shape=input_size,
                             strides=(stride, stride), padding="same"))
        else:
            model.add(Conv2D(conv_count, filter_width,
                             strides=(stride, stride), padding="same"))
        model.add(LeakyReLU(alpha=leaky_relu_alpha))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def stack_gen_disc(generator, discriminator):
    """
    Combines generator and discrminator layers together while freezing the weights of the discriminator layers

    Args:
        generator:
        discriminator:

    Returns:
        Generator layers attached to discriminator layers.
    """
    model = Sequential()
    for layer in generator.layers:
        model.add(layer)
    for layer in discriminator.layers:
        layer.trainable = False
        model.add(layer)
    return model


def stack_gen_encoder(generator, encoder):
    """
    Combines generator and encoder layers together while freezing the weights of the generator layers.
    This is used to train the encoder network to convert image data into a low-dimensional vector
     representation.

    Args:
        generator: Decoder network
        encoder: Encoder network
    Returns:
        Encoder layers attached to generator layers
    """
    model = Sequential()
    for layer in generator.layers:
        layer.trainable = False
        model.add(layer)
    for layer in encoder.layers:
        layer.trainable = True
        model.add(layer)
    return model


def train_gan(train_data, generator, discriminator, gan_path, gan_index, batch_size=128, num_epochs=(10, 100, 1000),
              gen_optimizer="adam", disc_optimizer="adam", gen_input_size=100,
              gen_loss="binary_crossentropy", disc_loss="binary_crossentropy", metrics=("accuracy", ),
              encoder=None, encoder_loss="mean_squared_error", min_vals=(0, 0, 0), max_vals=(255, 255, 255),
              out_dtype="float32"):
    """
    Train generative adversarial network

    Args:
        train_data:
        generator:
        discriminator:
        batch_size:
        num_epochs:
        gen_optimizer:
        disc_optimizer:
        gen_loss:
        disc_loss:
        metrics:
        encoder:
        encoder_loss:
        min_vals:
        max_vals:
    Returns:

    """
    metrics = list(metrics)
    batch_size = int(batch_size)
    batch_half = int(batch_size // 2)
    generator.compile(optimizer=gen_optimizer, loss=gen_loss, metrics=metrics)
    print(generator.summary())
    discriminator.compile(optimizer=disc_optimizer, loss=disc_loss, metrics=metrics)
    print(discriminator.summary())
    gen_on_disc = stack_gen_disc(generator, discriminator)
    gen_on_disc.compile(optimizer=gen_optimizer, loss=gen_loss, metrics=metrics)
    gen_on_encoder = None
    if encoder is not None:
        encoder.compile(optimizer=gen_optimizer, loss=encoder_loss)
        gen_on_encoder = stack_gen_encoder(generator, encoder)
        gen_on_encoder.compile(optimizer=gen_optimizer, loss=encoder_loss)
    train_order = np.arange(train_data.shape[0])
    gen_loss_history = []
    disc_loss_history = []
    encoder_loss_history = []
    current_epoch = []
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]))
    batch_labels = np.zeros(batch_size, dtype=int)
    batch_labels[:batch_half] = 1
    gen_labels = np.ones(batch_size, dtype=int)
    for epoch in range(1, max(num_epochs) + 1):
        np.random.shuffle(train_order)
        for b, b_index in enumerate(np.arange(batch_half, train_data.shape[0] + batch_half, batch_half)):
            disc_noise = np.random.uniform(-1, 1, size=(batch_size, gen_input_size))
            gen_noise = np.random.uniform(-1, 1, size=(batch_size, gen_input_size))
            combo_data_batch[batch_half:] = generator.predict_on_batch(disc_noise)[::2]
            combo_data_batch[:batch_half] = train_data[train_order[b_index - batch_half: b_index]]
            print("{3} Train Discriminator Combo: {0} Epoch: {1} Batch: {2}".format(gan_index,
                                                                                    epoch,
                                                                                    b,
                                                                                    pd.Timestamp("now")))
            disc_loss_history.append(discriminator.train_on_batch(combo_data_batch, batch_labels))
            print("{3} Train Generator Combo: {0} Epoch: {1} Batch: {2}".format(gan_index,
                                                                                epoch,
                                                                                b,
                                                                                pd.Timestamp("now")))
            gen_loss_history.append(gen_on_disc.train_on_batch(gen_noise, gen_labels))
            print("Disc Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index,
                                                                                                    epoch, b,
                                                                                                    *disc_loss_history[-1]))
            print("Gen Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index,
                                                                                                   epoch, b,
                                                                                                   *gen_loss_history[-1]))

            current_epoch.append((epoch,b))
            if encoder is not None:
                encoder_loss_history.append(gen_on_encoder.train_on_batch(gen_noise, gen_noise))
        if epoch in num_epochs:
            print("{2} Save Models Combo: {0} Epoch: {1}".format(gan_index,
                                                                 epoch,
                                                                 pd.Timestamp("now")))
            generator.save(join(gan_path, "gan_generator_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            discriminator.save(join(gan_path, "gan_discriminator_{0:06d}_{1:04d}.h5".format(gan_index, epoch)))
            gen_noise = np.random.uniform(-1, 1, size=(batch_size, gen_input_size))
            gen_data_epoch = unscale_multivariate_data(generator.predict_on_batch(gen_noise), min_vals, max_vals)
            gen_da = xr.DataArray(gen_data_epoch.astype(out_dtype), coords={"p": np.arange(gen_data_epoch.shape[0]),
                                                          "y": np.arange(gen_data_epoch.shape[1]),
                                                          "x": np.arange(gen_data_epoch.shape[2]),
                                                          "channel": np.arange(train_data.shape[-1])},
                                  dims=("p", "y", "x", "channel"),
                                  attrs={"long_name": "Synthetic data", "units": ""})
            gen_da.to_dataset(name="gen_patch").to_netcdf(join(gan_path,
                                                               "gan_gen_patches_{0:06d}_epoch_{1:04d}.nc".format(gan_index, epoch)),
                                                          encoding={"gen_patch": {"zlib": True,
                                                                                  "complevel": 1}})
            if encoder is not None:
                encoder.save(join(gan_path, "gan_encoder_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
    hist_cols = ["Epoch", "Batch", "Disc Loss"] + ["Disc " + m for m in metrics] + \
                ["Gen Loss"] + ["Gen " + m for m in metrics] + ["Encoder Loss"]
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history, gen_loss_history, encoder_loss_history]),
                           columns=hist_cols)
    return history


def rescale_data(data):
    scaled_data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
    return scaled_data


def rescale_multivariate_data(data):
    scaled_data = np.zeros(data.shape)
    for i in range(data.shape[-1]):
        scaled_data[:, :, :, i] = rescale_data(data[:, :, :, i])
    return scaled_data


def unscale_data(data, min_val=0, max_val=255):
    unscaled_data = (data + 1) / 2 * (max_val - min_val) + min_val
    return unscaled_data


def unscale_multivariate_data(data, min_vals, max_vals):
    unscaled_data = np.zeros(data.shape)
    for i in range(data.shape[-1]):
        unscaled_data[:, :, :, i] = unscale_data(data[:, :, :, i],
                                                 min_vals[i],
                                                 max_vals[i])
    return unscaled_data