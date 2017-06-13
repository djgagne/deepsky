from __future__ import division
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Input, Conv1D
from keras.layers import Activation, Reshape, LeakyReLU, concatenate, Dropout
import xarray as xr
from os.path import join
import keras.backend as K


def generator_model(input_size=100, filter_width=5, min_data_width=4,
                    min_conv_filters=64, output_size=(32, 32, 1), stride=2):
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

    Returns:
        Keras convolutional neural network.
    """
    num_layers = int(np.log2(output_size[0]) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers - 1))
    curr_conv_filters = max_conv_filters
    vector_input = Input(shape=(input_size, ), name="gen_input")
    model = Dense(units=max_conv_filters * min_data_width * min_data_width)(vector_input)
    model = Reshape((min_data_width, min_data_width, max_conv_filters))(model)
    model = Activation("elu")(model)
    for i in range(1, num_layers):
        curr_conv_filters //= 2
        model = Conv2DTranspose(curr_conv_filters, filter_width,
                                  strides=(stride, stride), padding="same")(model)
        model = Activation("elu")(model)
    model = Conv2DTranspose(output_size[-1], filter_width,
                              strides=(stride, stride),
                              padding="same")(model)
    model = Activation("tanh")(model)
    return model, vector_input


def encoder_model(input_size=(32, 32, 1), filter_width=5, min_data_width=4,
                    min_conv_filters=64, output_size=100, stride=2):
    """
    Creates an encoder convolutional neural network that reproduces the generator input vector. The keyword arguments
    allow aspects of the structure of the generator to be tuned for optimal performance.

    Args:
        input_size (tuple of ints): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the last convolved layer
        min_conv_filters (int): Number of convolutional filters in the first convolutional layer
        output_size (int): Dimensions of the output
        stride (int): Number of pixels that the convolution filter shifts between operations.

    Returns:
        Keras convolutional neural network.
    """
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    image_input = Input(shape=input_size, name="enc_input")
    model = None
    for c in range(num_layers):
        if c == 0:
            model = Conv2D(curr_conv_filters, filter_width,
                           strides=(stride, stride), padding="same")(image_input)
        else:
            model = Conv2D(curr_conv_filters, filter_width,
                           strides=(stride, stride), padding="same")(model)
        model = Activation("elu")(model)
        curr_conv_filters *= 2
    model = Flatten()(model)
    model = Dense(output_size)(model)
    model = Activation("tanh")(model)
    return model, image_input


def encoder_disc_model(input_size=(32, 32, 1), filter_width=5, min_data_width=4,
                       min_conv_filters=64, output_size=100, stride=2):
    """
    Creates an encoder convolutional neural network that reproduces the generator input vector. The keyword arguments
    allow aspects of the structure of the generator to be tuned for optimal performance.

    Args:
        input_size (tuple of ints): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the last convolved layer
        min_conv_filters (int): Number of convolutional filters in the first convolutional layer
        output_size (int): Dimensions of the output
        stride (int): Number of pixels that the convolution filter shifts between operations.

    Returns:
        Keras convolutional neural network.
    """
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    image_input = Input(shape=input_size, name="enc_input")
    model = image_input
    for c in range(num_layers):
        model = Conv2D(curr_conv_filters, filter_width,
                       strides=(stride, stride), padding="same")(model)
        model = Activation("elu")(model)
        curr_conv_filters *= 2
    model = Flatten()(model)
    enc_model = Dense(output_size)(model)
    enc_model = Activation("tanh")(enc_model)
    disc_model = Dense(1)(model)
    disc_model = Activation("sigmoid")(disc_model)
    return disc_model, enc_model, image_input


def discriminator_model(input_size=(32, 32, 1), stride=2, filter_width=5,
                        min_conv_filters=64, min_data_width=4, leaky_relu_alpha=0.2):
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
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    model = Sequential()
    for c in range(num_layers):
        if c == 0:
            model.add(Conv2D(curr_conv_filters, filter_width, input_shape=input_size,
                             strides=(stride, stride), padding="same"))
        else:
            model.add(Conv2D(curr_conv_filters, filter_width,
                             strides=(stride, stride), padding="same"))
        model.add(LeakyReLU(alpha=leaky_relu_alpha))
        curr_conv_filters *= 2
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def joint_discriminator_model(gen_model, enc_model, image_input, vector_input, stride=2, filter_width=5,
                              min_conv_filters=64, min_data_width=4, leaky_relu_alpha=0.2, num_vec_layers=2, vec_conv_filters=128):
    """
    Creates a discriminator model for a generative adversarial network.

    Args:
        gen_model (keras layers): Connected generator model layers before they are input into Model
        enc_model (keras layers): Connected encoder model layers before they are input into Model
        image_input (keras Input layer): Input layer for image-like data
        vector_input (keras Input layer): Input layer for low-dimensional vector
        stride (int): Number of pixels the convolution filter is shifted between operations
        filter_width (int): Width of convolution filters
        min_conv_filters (int): Number of convolution filters in the first layer. Doubles in each subsequent layer
        min_data_width (int): Smallest width of input data after convolution downsampling before flattening
        leaky_relu_alpha (float): scaling coefficient for negative values in Leaky Rectified Linear Unit

    Returns:
        Combined generator, encoder, and discriminator; Discriminator
    """
    image_input_size = image_input.shape.as_list()[1:]
    vector_input_size = vector_input.shape.as_list()[1:]
    num_layers = int(np.log2(image_input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers - 1))
    vec_layer_list = list()
    vec_layer_list.append(Dense(int(np.prod(image_input_size[:-1]))))
    vec_layer_list.append(Activation("tanh"))
    vec_layer_list.append(Reshape((image_input_size[0], image_input_size[1], 1)))
    conv_layer_list = list()
    for c in range(num_layers):
        if c == 0:
            conv_layer_list.append(Conv2D(curr_conv_filters, filter_width,
                                          strides=(stride, stride), padding="same", name="image_first"))
            conv_layer_list.append(LeakyReLU(alpha=leaky_relu_alpha))
        else:
            conv_layer_list.append(Conv2D(curr_conv_filters, filter_width,
                                          strides=(stride, stride), padding="same"))
            conv_layer_list.append(LeakyReLU(alpha=leaky_relu_alpha))
        curr_conv_filters *= 2
    conv_layer_list.append(Flatten())
    combined_layer_list = list()
    combined_layer_list.append(Dense(1))
    combined_layer_list.append(Activation("sigmoid"))
    full_model = gen_model
    full_model_vec = vec_layer_list[0](enc_model)
    for vec_layer in vec_layer_list[1:]:
        full_model_vec = vec_layer(full_model_vec)
    full_model = concatenate([full_model, full_model_vec])
    for conv_layer in conv_layer_list:
        full_model = conv_layer(full_model)
    for combined_layer in combined_layer_list:
        full_model = combined_layer(full_model)
    full_model_obj = Model([image_input, vector_input], full_model)
    disc_model = image_input
    disc_model_vec = vec_layer_list[0](vector_input)
    for vec_layer in vec_layer_list[1:]:
        disc_model_vec = vec_layer(disc_model_vec)
    disc_model = concatenate([disc_model, disc_model_vec])
    for conv_layer in conv_layer_list:
        disc_model = conv_layer(disc_model)
    for combined_layer in combined_layer_list:
        disc_model = combined_layer(disc_model)
    disc_model_obj = Model([image_input, vector_input], disc_model)
    return full_model_obj, disc_model_obj


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


def stack_gen_encoder(generator, encoder, discriminator):
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
        if layer in discriminator.layers:
            layer.trainable = False
        model.add(layer)
    return model


def stack_enc_gen(encoder, generator, discriminator):
    """
    Combines encoder and generator layers together while freezing the weights of all layers except the last
    in the encoder. This is used to train the encoder network to convert image data into a low-dimensional vector
     representation.

    Args:
        encoder: Encoder network
        generator: Decoder network
        discriminator: Discriminator network. Used to freeze shared layers in encoder
    Returns:
        Encoder layers attached to generator layers
    """
    model = Sequential()
    for layer in encoder.layers:
        if layer in discriminator.layers:
            layer.trainable = False
        model.add(layer)
    for layer in generator.layers:
        layer.trainable = False
        model.add(layer)
    return model


def stack_encoder_gen_disc(encoder, generator, discriminator):
    model = Sequential()
    for layer in encoder.layers:
        model.add(layer)
    for layer in generator.layers:
        layer.trainable = False
        model.add(layer)
    for layer in discriminator.layers:
        layer.trainable = False
        model.add(layer)
    return model


def train_linked_gan(train_data, generator, encoder, discriminator, gen_disc, enc_gen, vec_size, gan_path, gan_index,
                     metrics=("accuracy", ), batch_size=128, num_epochs=(1, 5, 10), min_vals=(0, 0, 0),
                     max_vals=(255, 255, 255), out_dtype="uint8"):
    batch_size = int(batch_size)
    batch_half = int(batch_size // 2)
    train_order = np.arange(train_data.shape[0])
    disc_loss_history = []
    gen_loss_history = []
    gen_enc_loss_history = []
    time_history = []
    current_epoch = []
    batch_labels = np.zeros(batch_size, dtype=int)
    batch_labels[:batch_half] = 1
    disc_labels = np.zeros(batch_size, dtype=int)
    gen_labels = np.ones(batch_size, dtype=int)
    batch_vec = np.zeros((batch_size, vec_size))
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]))
    hist_cols = ["Epoch", "Batch", "Disc Loss"] + ["Disc " + m for m in metrics] + \
                ["Gen Loss"] + ["Gen " + m for m in metrics] + ["Gen_Enc Loss"]
    for epoch in range(1, max(num_epochs) + 1):
        np.random.shuffle(train_order)
        for b, b_index in enumerate(np.arange(batch_half, train_data.shape[0] + batch_half, batch_half)):
            disc_labels[:] = batch_labels[:]
            #label_switches = np.random.binomial(1, 0.05, size=(batch_size))
            #disc_labels[label_switches == 1] = 1 - disc_labels[label_switches == 1]
            batch_vec[:] = np.random.uniform(-1, 1, size=(batch_size, vec_size))
            combo_data_batch[:batch_half] = train_data[train_order[b_index - batch_half: b_index]]
            combo_data_batch[batch_half:] = generator.predict_on_batch(batch_vec[batch_half:])
            disc_loss_history.append(discriminator.train_on_batch(combo_data_batch, disc_labels))
            print("Disc Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index, 
                                                                                                    epoch, b,
                                                                                                    *disc_loss_history[-1]))
            gen_loss_history.append(gen_disc.train_on_batch(batch_vec,
                                                            gen_labels))
            print("Gen Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index, 
                                                                                                        epoch, b,
                                                                                                        *gen_loss_history[-1]))
            gen_enc_loss_history.append(enc_gen.train_on_batch(combo_data_batch, combo_data_batch))
            print("Gen Enc Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}".format(gan_index, 
                                                                                   epoch, b,
                                                                                   gen_enc_loss_history[-1]))
            time_history.append(pd.Timestamp("now"))
            current_epoch.append((epoch, b))
        if epoch in num_epochs:
            print("{2} Save Models Combo: {0} Epoch: {1}".format(gan_index,
                                                                 epoch,
                                                                 pd.Timestamp("now")))
            generator.save(join(gan_path, "gan_generator_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            discriminator.save(join(gan_path, "gan_discriminator_{0:06d}_{1:04d}.h5".format(gan_index, epoch)))
            gen_noise = np.random.uniform(-1, 1, size=(batch_size, vec_size))
            gen_data_epoch = unscale_multivariate_data(generator.predict_on_batch(gen_noise), min_vals, max_vals)
            gen_da = xr.DataArray(gen_data_epoch.astype(out_dtype), coords={"p": np.arange(gen_data_epoch.shape[0]),
                                                                            "y": np.arange(gen_data_epoch.shape[1]),
                                                                            "x": np.arange(gen_data_epoch.shape[2]),
                                                                            "channel": np.arange(train_data.shape[-1])},
                                  dims=("p", "y", "x", "channel"),
                                  attrs={"long_name": "Synthetic data", "units": ""})
            gen_da.to_dataset(name="gen_patch").to_netcdf(join(gan_path,
                                                               "gan_gen_patches_{0:03d}_epoch_{1:03d}.nc".format(
                                                                   gan_index, epoch)),
                                                          encoding={"gen_patch": {"zlib": True,
                                                                                  "complevel": 1}})
            encoder.save(join(gan_path, "gan_encoder_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
        time_history_index = pd.DatetimeIndex(time_history)
        history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history,
                                            gen_loss_history, np.array(gen_enc_loss_history).reshape(-1, 1)]),
                                index=time_history_index, columns=hist_cols)
        history.to_csv(join(gan_path, "gan_loss_history_{0:03d}.csv".format(gan_index)), index_label="Time")
    time_history_index = pd.DatetimeIndex(time_history)
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history,
                                      gen_loss_history, 
                                      np.array(gen_enc_loss_history).reshape(-1, 1)]),
                           index=time_history_index, columns=hist_cols)
    history.to_csv(join(gan_path, "gan_loss_history_{0:03d}.csv".format(gan_index)), index_label="Time")
    return history


def train_full_gan(train_data, generator, encoder, discriminator, combined_model, vec_size, gan_path, gan_index,
                   metrics=("accuracy", ),
                   batch_size=128, num_epochs=(1, 5, 10), min_vals=(0, 0, 0), max_vals=(255, 255, 255),
                   out_dtype="uint8"):
    """
    Train GAN model that contains 3 networks with the discriminator receiving joint information from the generator and
    encoder simultaneously.
    
    Args:
        train_data: Numpy array of gridded data
        generator: Compiled generator network object
        encoder: Compiled encoder network object
        discriminator: Compiled discriminator network object
        combined_model: Compiled combination of encoder, generator, and discriminator where the discriminator
            weights are frozen.
        vec_size: Size of the encoded vector (input of generator and output of encoder)
        gan_path: Path to where GAN model and log files are saved
        gan_index: GAN Configuration number 
        metrics: Additional metrics to track during GAN training
        batch_size: Number of training examples per mini-batch
        num_epochs: Epochs when models are saved to disk
        min_vals: Minimum values of each training data dimension for scaling purposes
        max_vals: Maximum values of each training data dimension for scaling purposes
        out_dtype: Datatype of synthetic examples

    Returns:

    """
    batch_size = int(batch_size)
    batch_half = int(batch_size // 2)
    train_order = np.arange(train_data.shape[0])
    disc_loss_history = []
    combined_loss_history = []
    time_history = []
    current_epoch = []
    batch_labels = np.zeros(batch_size, dtype=int)
    batch_labels[:batch_half] = 1
    batch_vec = np.zeros((batch_size, vec_size))
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]))
    hist_cols = ["Epoch", "Batch", "Disc Loss"] + ["Disc " + m for m in metrics] + \
                ["Gen Loss"] + ["Gen " + m for m in metrics] 
    for epoch in range(1, max(num_epochs) + 1):
        np.random.shuffle(train_order)
        for b, b_index in enumerate(np.arange(batch_half, train_data.shape[0] + batch_half, batch_half)):
            batch_vec[:batch_half] = encoder.predict_on_batch(train_data[train_order[b_index - batch_half: b_index]])
            batch_vec[batch_half:] = np.random.uniform(-1, 1, size=(batch_half, vec_size))
            combo_data_batch[:batch_half] = train_data[train_order[b_index - batch_half: b_index]]
            combo_data_batch[batch_half:] = generator.predict_on_batch(batch_vec[batch_half:])
            print("{3} Train Discriminator Combo: {0} Epoch: {1} Batch: {2}".format(gan_index,
                                                                                    epoch,
                                                                                    b,
                                                                                    pd.Timestamp("now")))
            
            #for l in discriminator.layers:
            #    weights = l.get_weights()
            #    weights = [np.clip(w, -0.1, 0.1) for w in weights]
            #    l.set_weights(weights)
            disc_loss_history.append(discriminator.train_on_batch([combo_data_batch, batch_vec], batch_labels)) 
            print("Disc Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index, 
                                                                                                    epoch, b,
                                                                                                    *disc_loss_history[-1]))
            print("{3} Train Gen/Encoder Combo: {0} Epoch: {1} Batch: {2}".format(gan_index,
                                                                                epoch,
                                                                                b,
                                                                                pd.Timestamp("now")))
            combined_loss_history.append(combined_model.train_on_batch([combo_data_batch, batch_vec],
                                                                       batch_labels[::-1]))
            print("Combined Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index, 
                                                                                                        epoch, b,
                                                                                                        *combined_loss_history[-1]))
            time_history.append(pd.Timestamp("now"))
            current_epoch.append((epoch,b))
        if epoch in num_epochs:
            print("{2} Save Models Combo: {0} Epoch: {1}".format(gan_index,
                                                                 epoch,
                                                                 pd.Timestamp("now")))
            generator.save(join(gan_path, "gan_generator_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            discriminator.save(join(gan_path, "gan_discriminator_{0:06d}_{1:04d}.h5".format(gan_index, epoch)))
            gen_noise = np.random.uniform(-1, 1, size=(batch_size, vec_size))
            gen_data_epoch = unscale_multivariate_data(generator.predict_on_batch(gen_noise), min_vals, max_vals)
            gen_da = xr.DataArray(gen_data_epoch.astype(out_dtype), coords={"p": np.arange(gen_data_epoch.shape[0]),
                                                          "y": np.arange(gen_data_epoch.shape[1]),
                                                          "x": np.arange(gen_data_epoch.shape[2]),
                                                          "channel": np.arange(train_data.shape[-1])},
                                  dims=("p", "y", "x", "channel"),
                                  attrs={"long_name": "Synthetic data", "units": ""})
            gen_da.to_dataset(name="gen_patch").to_netcdf(join(gan_path,
                                                               "gan_gen_patches_{0:03d}_epoch_{1:03d}.nc".format(gan_index, epoch)),
                                                          encoding={"gen_patch": {"zlib": True,
                                                                                  "complevel": 1}})
            encoder.save(join(gan_path, "gan_encoder_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            time_history_index = pd.DatetimeIndex(time_history)
            history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history, combined_loss_history]),
                                   index=time_history_index, columns=hist_cols)
            history.to_csv(join(gan_path, "gan_loss_history_{0:03d}.csv".format(gan_index)), index_label="Time")
    time_history_index = pd.DatetimeIndex(time_history)
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history, combined_loss_history]),
                           index=time_history_index, columns=hist_cols)
    history.to_csv(join(gan_path, "gan_loss_history_{0:03d}.csv".format(gan_index)), index_label="Time")
    return


def train_gan(train_data, generator, discriminator, gan_path, gan_index, batch_size=128, num_epochs=(1, 5, 10),
              gen_optimizer="adam", disc_optimizer="adam", gen_input_size=100,
              gen_loss="binary_crossentropy", disc_loss="binary_crossentropy", metrics=("accuracy", ),
              encoder=None, encoder_loss="binary_crossentropy", min_vals=(0, 0, 0), max_vals=(255, 255, 255),
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
    encoder_on_gan = None
    if encoder is not None:
        encoder.compile(optimizer=gen_optimizer, loss=encoder_loss)
        encoder_on_gan = stack_encoder_gen_disc(encoder, generator, discriminator)
        encoder_on_gan.compile(optimizer=gen_optimizer, loss=encoder_loss)
    train_order = np.arange(train_data.shape[0])
    gen_loss_history = []
    disc_loss_history = []
    encoder_loss_history = []
    time_history = []
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
                encoder_loss_history.append(encoder_on_gan.train_on_batch(combo_data_batch, gen_labels))
            time_history.append(pd.Timestamp("now"))
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
    encoder_history = np.array(encoder_loss_history)
    encoder_history = encoder_history.reshape(-1, 1)
    time_history = pd.DatetimeIndex(time_history)
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history, gen_loss_history, encoder_history]),
                           index=time_history, columns=hist_cols)
    return history


def wgan(y_true, y_pred):
    real_score = K.sum(y_pred * y_true) / K.sum(y_true)
    fake_score = K.sum(y_pred * K.reverse(y_true, 0)) / K.sum(y_true)
    return fake_score - real_score


def gan_loss(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    zeros = K.zeros_like(y_true_flat)
    ones = K.ones_like(y_true_flat)
    switched = K.tf.where(K.equal(y_true_flat, zeros), ones - y_pred_flat, y_pred_flat)
    switched_nonzero = K.tf.where(K.equal(switched, zeros), ones * 0.001, switched)
    return -K.mean(K.log(switched_nonzero))


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
