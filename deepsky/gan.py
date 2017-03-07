from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Deconvolution2D, Flatten, Dense, BatchNormalization
from keras.layers import Activation, Reshape, LeakyReLU


def generator_model(batch_size=128, input_size=100, filter_width=5, min_data_width=4,
                    max_conv_filters=256, output_size=(32, 32, 1), stride=2):
    """
    Creates a generator convolutional neural network for a generative adversarial network set. The keyword arguments
    allow aspects of the structure of the generator to be tuned for optimal performance.

    Args:
        batch_size (int): Number of examples in each batch. The Deconvoluational layers require a fixed batch size
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
    while data_widths[-1] <= output_size[0] // stride:
        data_widths.append(data_widths[-1] * stride)
        conv_filters.append(conv_filters[-1] // stride)
    model = Sequential()
    model.add(Dense(input_shape=(input_size, ), output_dim=max_conv_filters * min_data_width * min_data_width))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Reshape((min_data_width, min_data_width, max_conv_filters)))
    for i in range(1, len(data_widths)):
        model.add(Deconvolution2D(conv_filters[i], filter_width, filter_width,
                                  output_shape=(batch_size, data_widths[i], data_widths[i], conv_filters[i]),
                                  subsample=(stride, stride), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
    model.add(Deconvolution2D(output_size[-1], filter_width, filter_width,
                              output_shape=(batch_size, output_size[0], output_size[1], output_size[2]),
                              subsample=(stride, stride),
                              border_mode="same"))
    model.add(Activation("tanh"))
    return model


def discriminator_model(input_size=(32, 32, 1), stride=2, filter_width=5,
                        min_conv_filters=16, min_data_width=4, leaky_relu_alpha=0.2):
    """
    Creates a discriminator model for a generative adversarial network.

    Args:
        input_size (tuple of size 3): Dimensions of input data
        stride (int): Number of pixels the convolution filter is shifted between operations
        filter_width (int): Width of convolution filters
        min_conv_filters (int): Number of convolution filters in the first layer. Doubles in each subsequent layer
        min_data_width (int): Smallest width of input data after convolution downsampling before flattening
        leaky_relu_alpha (float): scaling coefficient for negative values in Leaky Rectified Linear Unit

    Returns:
        Keras generator model
    """
    curr_width = input_size[0]
    conv_filters = [min_conv_filters]
    while curr_width > min_data_width:
        conv_filters.append(conv_filters[-1] * 2)
        curr_width //= stride
    model = Sequential()
    for conv_count in conv_filters:
        model.add(Convolution2D(conv_count, filter_width, filter_width, input_shape=input_size,
                                subsample=(stride, stride), border_mode="same"))
        model.add(BatchNormalization())
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


def train_gan(train_data, generator, discriminator, batch_size, num_epochs,
              gen_optimizer, disc_optimizer, gen_loss, disc_loss, metrics):
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

    Returns:

    """
    batch_half = int(batch_size // 2)
    generator.compile(optimizer=gen_optimizer, loss=gen_loss, metrics=metrics)
    discriminator.compile(optimizer=disc_optimizer, loss=disc_loss, metrics=metrics)
    gen_on_disc = stack_gen_disc(generator, discriminator)
    gen_on_disc.compile(optimizer=gen_optimizer, loss=gen_loss, metrics=metrics)
    train_order = np.arange(train_data.shape[0])
    gen_loss_history = []
    disc_loss_history = []
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]))
    batch_labels = np.zeros(batch_size)
    batch_labels[:batch_half] = 1
    gen_labels = np.ones(batch_size)
    for epoch in range(num_epochs):
        np.random.shuffle(train_order)
        for b_index in np.arange(batch_half, train_data.shape[0] + batch_half, batch_half):
            disc_noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            gen_noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            combo_data_batch[batch_half:] = generator.test_on_batch(disc_noise)[::2]
            combo_data_batch[:batch_half] = train_data[train_order[b_index - batch_half: b_index]]
            disc_loss_history.append(discriminator.train_on_batch(combo_data_batch, batch_labels))
            gen_loss_history.append(gen_on_disc.train_on_batch(gen_noise, gen_labels))
    return np.array(disc_loss_history), np.array(gen_loss_history)


def rescale_data(data):
    scaled_data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
    return scaled_data
