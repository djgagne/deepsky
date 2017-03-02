import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Deconvolution2D, Flatten, Dense, BatchNormalization
from keras.layers import Activation, Reshape, LeakyReLU, UpSampling2D
from keras.layers.core import K
from keras.optimizers import Adam


def generator_model(batch_size=128, input_size=100, filter_width=5, min_data_width=4,
                    max_conv_filters=256, output_size=(32, 32, 1), stride=2):
    data_widths = [min_data_width]
    conv_filters = [max_conv_filters]
    while data_widths[-1] <= output_size[0] // 2:
        data_widths.append(data_widths[-1] * 2)
        conv_filters.append(conv_filters[-1] // 2)
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
    curr_width = input_size[0]
    conv_filters = [min_conv_filters]
    while curr_width > min_data_width:
        conv_filters.append(conv_filters[-1] * 2)
        curr_width = curr_width // 2
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


def gen_disc_stack(generator, discriminator):
    model = Sequential()
    for layer in generator.layers:
        model.add(layer)
    for layer in discriminator.layers:
        layer.trainable = False
        model.add(layer)
    return model
