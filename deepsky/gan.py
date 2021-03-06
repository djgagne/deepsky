from __future__ import division
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, save_model
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Input, UpSampling2D, MaxPool2D, BatchNormalization
from keras.layers import Activation, Reshape, LeakyReLU, concatenate, Dropout, GaussianNoise, AveragePooling2D
from keras.regularizers import l2
import xarray as xr
from os.path import join
import keras.backend as K


def generator_model(input_size=100, filter_width=5, min_data_width=4,
                    min_conv_filters=64, output_size=(32, 32, 1), stride=2, activation="relu",
                    use_dropout=False, dropout_alpha=0,
                    use_noise=False, noise_sd=0.1, l2_reg=0.001):
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
        use_dropout (bool): Whether to use Dropout layers or not.
        dropout_alpha (float): proportion of nodes dropped out.
        use_noise: Whether to use a Gaussian noise layer after a convolution.
        noise_sd: Standard deviation of the Gaussian noise.

    Returns:
        Model output graph, model input
    """
    num_layers = int(np.log2(output_size[0]) - np.log2(min_data_width))
    max_conv_filters = int(min_conv_filters * 2 ** (num_layers))
    curr_conv_filters = max_conv_filters
    vector_input = Input(shape=(input_size, ), name="gen_input")
    model = Dense(units=max_conv_filters * min_data_width * min_data_width,
                  kernel_regularizer=l2(l2_reg), use_bias=False)(vector_input)
    model = Reshape((min_data_width, min_data_width, max_conv_filters))(model)
    if activation == "leaky":
        model = LeakyReLU(alpha=0.2)(model)
    else:
        model = Activation(activation)(model)
    for i in range(num_layers):
        curr_conv_filters //= 2
        model = Conv2DTranspose(curr_conv_filters, (filter_width, filter_width),
                                strides=(stride, stride), padding="same", kernel_regularizer=l2(l2_reg))(model)
        if activation == "leaky":
            model = LeakyReLU(alpha=0.2)(model)
        else:
            model = Activation(activation)(model)
        if use_dropout:
            model = Dropout(dropout_alpha)(model)
        if use_noise:
            model = GaussianNoise(noise_sd)(model)
        if stride == 1:
            model = UpSampling2D()(model)
    model = Conv2DTranspose(output_size[-1], (filter_width, filter_width),
                            strides=(1, 1),
                            padding="same", kernel_regularizer=l2(l2_reg))(model)
    model = BatchNormalization()(model)
    model_out = Model(vector_input, model)
    return model_out


def encoder_model(input_size=(32, 32, 1), filter_width=5, min_data_width=4,
                  min_conv_filters=64, output_size=100, stride=2, activation="relu", output_activation="linear",
                  use_dropout=False, dropout_alpha=0, use_noise=False, noise_sd=0.1, pooling="mean"):
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
        activation (str): Type of activation used for convolutional layers. Use "leaky" for Leaky ReLU.
        output_activation (str): Type of activation used on the output layer
        use_dropout (bool): Whether to use Dropout layers or not.
        dropout_alpha (float): proportion of nodes dropped out.
        use_noise (bool): Whether to use a Gaussian noise layer after a convolution.
        noise_sd (float): Standard deviation of the Gaussian noise.
        pooling (str): Type of pooling to use if stride=1. Options: "mean" or "max".
    Returns:
        Keras convolutional neural network.
    """
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    image_input = Input(shape=input_size, name="enc_input")
    model = None
    for c in range(num_layers):
        if c == 0:
            model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                           strides=(stride, stride), padding="same", kernel_regularizer=l2())(image_input)
        else:
            model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                           strides=(stride, stride), padding="same", kernel_regularizer=l2())(model)
        if activation == "leaky":
            model = LeakyReLU(0.2)(model)
        else:
            model = Activation(activation)(model)
        if use_dropout:
            model = Dropout(dropout_alpha)(model)
        if use_noise:
            model = GaussianNoise(noise_sd)(model)
        if stride == 1:
            if pooling.lower() == "mean":
                model = AveragePooling2D()(model)
            else:
                model = MaxPool2D()(model)
        curr_conv_filters *= 2
    model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                       strides=(1, 1), padding="same", kernel_regularizer=l2())(model)
    model = Flatten()(model)
    model = Dense(output_size)(model)
    model = BatchNormalization()(model)
    model_out = Model(image_input, model)
    return model_out


def encoder_disc_model(input_size=(32, 32, 1), filter_width=5, min_data_width=4,
                       min_conv_filters=64, output_size=100, stride=2, activation="relu",
                       encoder_output_activation="linear",
                       use_dropout=False, dropout_alpha=0, use_noise=False, noise_sd=0.1, pooling="mean"):
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
        use_dropout (bool): Whether to use Dropout layers or not.
        dropout_alpha (float): proportion of nodes dropped out.
        use_noise (bool): Whether to use a Gaussian noise layer after a convolution.
        noise_sd (float): Standard deviation of the Gaussian noise.
        pooling (str): Type of pooling to use if stride=1. Options: "mean" or "max".

    Returns:
        discriminator model output, encoder model output, image input
    """
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    image_input = Input(shape=input_size, name="enc_input")
    model = image_input
    for c in range(num_layers):
        model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                       strides=(stride, stride), padding="same")(model)
        if activation == "leaky":
            model = LeakyReLU(0.2)(model)
        else:
            model = Activation(activation)(model)
        if use_dropout:
            model = Dropout(dropout_alpha)(model)
        if use_noise:
            model = GaussianNoise(noise_sd)(model)
        if stride == 1:
            if pooling.lower() == "mean":
                model = AveragePooling2D()(model)
            else:
                model = MaxPool2D()(model)
        curr_conv_filters *= 2
    model = Flatten()(model)
    enc_model = Dense(int(0.5 * curr_conv_filters * filter_width ** 2))(model)
    if activation == "leaky":
        enc_model = LeakyReLU(0.2)(enc_model)
    else:
        enc_model = Activation(activation)(enc_model)
    enc_model = Dense(output_size)(enc_model)
    enc_model = Activation(encoder_output_activation)(enc_model)
    disc_model = Dense(1)(model)
    disc_model = Activation("sigmoid")(disc_model)
    disc = Model(image_input, disc_model)
    enc = Model(image_input, enc_model)
    return disc, enc


def discriminator_model(input_size=(32, 32, 1), stride=2, filter_width=5,
                        min_conv_filters=64, min_data_width=4, activation="relu",
                        use_dropout=False, dropout_alpha=0, use_noise=False, noise_sd=0,
                        pooling="mean"):
    """
    Creates an discriminator convolutional neural network that reproduces the generator input vector.
    The keyword arguments allow aspects of the structure of the discriminator to be tuned for optimal performance.

    Args:
        input_size (tuple of ints): Number of nodes in the input layer.
        filter_width (int): Width of each convolutional filter
        min_data_width (int): Width of the last convolved layer
        min_conv_filters (int): Number of convolutional filters in the first convolutional layer
        stride (int): Number of pixels that the convolution filter shifts between operations.
        activation (str): Type of activation used for convolutional layers. Use "leaky" for Leaky ReLU.
        use_dropout (bool): Whether to use Dropout layers or not.
        dropout_alpha (float): proportion of nodes dropped out.
        use_noise (bool): Whether to use a Gaussian noise layer after a convolution.
        noise_sd (float): Standard deviation of the Gaussian noise.
        pooling (str): Type of pooling to use if stride=1. Options: "mean" or "max".

    Returns:
        discriminator model output, encoder model output, image input
    """
    num_layers = int(np.log2(input_size[0]) - np.log2(min_data_width))
    curr_conv_filters = min_conv_filters
    image_input = Input(shape=input_size, name="enc_input")
    for c in range(num_layers):
        if c == 0:
            model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                       strides=(stride, stride), padding="same", kernel_regularizer=l2())(image_input)
        else:
            model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                       strides=(stride, stride), padding="same", kernel_regularizer=l2())(model)
        if activation == "leaky":
            model = LeakyReLU(0.2)(model)
        else:
            model = Activation(activation)(model)
        if use_dropout:
            model = Dropout(dropout_alpha)(model)
        if use_noise:
            model = GaussianNoise(noise_sd)(model)
        if stride == 1:
            if pooling.lower() == "mean":
                model = AveragePooling2D()(model)
            else:
                model = MaxPool2D()(model)
        curr_conv_filters *= 2
    model = Conv2D(curr_conv_filters, (filter_width, filter_width),
                   strides=(1, 1), padding="same", kernel_regularizer=l2())(model)
    model = Flatten()(model)
    disc_model = Dense(1)(model)
    disc_model = Activation("sigmoid")(disc_model)
    model_out = Model(image_input, disc_model)
    return model_out


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
    discriminator.trainable = False
    stacked_model = discriminator(generator.output)
    model_obj = Model(generator.input, stacked_model)
    return model_obj


def stack_gen_enc(generator, encoder):
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
    generator.trainable = False
    model = encoder(generator.output)
    model_obj = Model(generator.input, model)
    return model_obj


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
    model_in = encoder.input
    model = model_in
    for layer in encoder.layers[1:]:
        if layer in discriminator.layers:
            layer.trainable = False
        model = layer(model)
    for layer in generator.layers:
        layer.trainable = False
        model = layer(model)
    model_obj = Model(model_in, model)
    return model_obj


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


def predict_stochastic(neural_net):
    """
    Have the neural network make predictions with the Dropout layers on, resulting in stochastic behavior from the
    neural net itself.

    Args:
        neural_net:
        data:

    Returns:

    """
    input_layer = neural_net.input
    output = neural_net.output
    pred_func = K.function([input_layer, K.learning_phase()], [output])
    return pred_func


def train_gan_quiet(all_train_data, generator, discriminator, encoder, gen_disc, gen_enc, vec_size,
                    batch_size, num_epochs, gan_index, out_path, update_interval=1):
    batch_size = int(batch_size)
    batch_half = int(batch_size // 2)
    batch_diff = all_train_data.shape[0] % batch_size
    if batch_diff > 0:
        train_data = all_train_data[:-batch_diff]
    else:
        train_data = all_train_data
    print(train_data.shape)
    train_order = np.arange(train_data.shape[0])
    batch_labels = np.zeros(batch_size, dtype=np.float32)
    batch_labels[:batch_half] = 1
    gen_labels = np.ones(batch_size, dtype=np.float32)
    batch_vec = np.zeros((batch_size, vec_size))
    gen_batch_vec = np.zeros((batch_size, vec_size), dtype=train_data.dtype)
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]), dtype=np.float32)
    hist_cols = ["Time", "Epoch", "Batch", "Disc Loss", "Gen Loss"]
    hist_dict = {h:[] for h in hist_cols}
    gen_pred_func = predict_stochastic(generator)
    for epoch in range(1, np.max(num_epochs) + 1):
        np.random.shuffle(train_order)
        for b, b_index in enumerate(np.arange(batch_half, train_data.shape[0] + batch_half, batch_half)):
            hist_dict["Time"].append(pd.Timestamp("now"))
            batch_vec[:] = np.random.normal(size=(batch_size, vec_size))
            gen_batch_vec[:] = np.random.normal(size=(batch_size, vec_size))
            combo_data_batch[:batch_half] = train_data[train_order[b_index - batch_half: b_index]]
            #combo_data_batch[batch_half:] = generator.predict_on_batch(batch_vec[batch_half:])
            combo_data_batch[batch_half:] = gen_pred_func([batch_vec[batch_half:], 1])[0]
            hist_dict["Epoch"].append(epoch)
            hist_dict["Batch"].append(b)
            hist_dict["Disc Loss"].append(discriminator.train_on_batch(combo_data_batch, batch_labels))
            hist_dict["Gen Loss"].append(gen_disc.train_on_batch(gen_batch_vec,
                                                            gen_labels))
            if b % update_interval == 0:
                print("Combo: {0:04d} Epoch: {1:02d} Batch: {2:03d} Disc: {3:0.3f} Gen: {4:0.3f}".format(gan_index,
                                                                                                         epoch, b,
                                                                                                         hist_dict["Disc Loss"][-1],
                                                                                                         hist_dict["Gen Loss"][-1]))
        if epoch in num_epochs:
            save_model(generator,
                       join(out_path, "gen_model_index_{0:04d}_epoch_{1:02d}.h5".format(gan_index, epoch)))
            save_model(discriminator,
                       join(out_path, "disc_model_index_{0:04d}_epoch_{1:02d}.h5".format(gan_index, epoch)))
    gen_inputs = np.random.normal(size=(train_data.shape[0], vec_size))
    print("Fit Encoder Combo: {0}".format(gan_index))
    gen_enc.fit(gen_inputs, gen_inputs, epochs=num_epochs[-1], batch_size=batch_size, verbose=2)
    save_model(encoder, join(out_path, "enc_model_index_{0:04d}_epoch_{1:02d}.h5".format(gan_index, num_epochs[-1])))
    time_history_index = pd.DatetimeIndex(hist_dict["Time"])
    history = pd.DataFrame(hist_dict,
                           index=time_history_index, columns=hist_cols)
    return history



def train_linked_gan(train_data, generator, encoder, discriminator, gen_disc, gen_enc, vec_size, gan_path, gan_index,
                     metrics=("accuracy", ), batch_size=128, num_epochs=(1, 5, 10), scaling_values=None,
                     out_dtype="uint8", ind_encoder=None):
    """
    Train GAN with encoder layers linked in. Also trains independent encoder on output from final epoch of generator.

    Args:
        train_data: 4D array of data used for training the model.
        generator: Generator neural network z->x
        encoder: Encoder neural network x->z.
        discriminator: Discriminator neural network x->p(x is real).
        gen_disc: Stacked generator and discriminator.
        gen_enc: Stacked generator and encoder.
        vec_size: Size of generator input vector
        gan_path: Path where GAN models and loss information are saved.
        gan_index: GAN configuration number.
        metrics: List of metrics used in discriminator
        batch_size: Number of training examples in each batch
        num_epochs: List of epochs at which models and generator examples are saved.
        scaling_values: pandas DataFrame of values used to normalize and rescale training data
        out_dtype: dtype of output data
        ind_encoder: Independent Encoder model
    Returns:
        None
    """
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
    gen_labels = np.ones(batch_size, dtype=int)
    batch_vec = np.zeros((batch_size, vec_size))
    gen_batch_vec = np.zeros((batch_size, vec_size), dtype=train_data.dtype)
    enc_batch_vec = np.zeros((batch_size, vec_size), dtype=train_data.dtype)
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]))
    hist_cols = ["Epoch", "Batch", "Disc Loss"] + ["Disc " + m for m in metrics] + \
                ["Gen Loss"] + ["Gen " + m for m in metrics] + ["Enc Loss"] + ["Enc " + m for m in ["mse", "mae"]]
    batch_divs = np.arange(batch_half, train_data.shape[0] + batch_half, batch_half)
    print(batch_divs, train_data.shape, gan_index)
    for epoch in range(1, max(num_epochs) + 1):
        np.random.shuffle(train_order)
        for b, b_index in enumerate(np.arange(batch_half, train_data.shape[0] + batch_half, batch_half)):
            batch_vec[:] = np.random.normal(size=(batch_size, vec_size))
            gen_batch_vec[:] = np.random.normal(size=(batch_size, vec_size))
            enc_batch_vec[:] = np.random.normal(size=(batch_size, vec_size))
            combo_data_batch[:batch_half] = train_data[train_order[b_index - batch_half: b_index]]
            combo_data_batch[batch_half:] = generator.predict_on_batch(batch_vec[batch_half:])
            disc_loss_history.append(discriminator.train_on_batch(combo_data_batch, batch_labels))
            print("Disc Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index, 
                                                                                                    epoch, b,
                                                                                                    *disc_loss_history[-1]))
            if b == 0:
                print(discriminator.summary())
                print(gen_disc.summary())
            gen_loss_history.append(gen_disc.train_on_batch(gen_batch_vec,
                                                            gen_labels))
            print("Gen Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}, Accuracy: {4:0.5f}".format(gan_index, 
                                                                                                   epoch, b,
                                                                                                   *gen_loss_history[-1]))
            if b == 0:
                print(generator.summary())
                print(gen_enc.summary())
            gen_enc_loss_history.append(gen_enc.train_on_batch(enc_batch_vec, enc_batch_vec))
            print("Gen Enc Combo: {0} Epoch: {1} Batch: {2} Loss: {3:0.5f}".format(gan_index, 
                                                                                   epoch, b,
                                                                                   gen_enc_loss_history[-1][0]))
            time_history.append(pd.Timestamp("now"))
            current_epoch.append((epoch, b))
        if epoch in num_epochs:
            print("{2} Save Models Combo: {0} Epoch: {1}".format(gan_index,
                                                                 epoch,
                                                                 pd.Timestamp("now")))
            generator.save(join(gan_path, "gan_generator_{0:04d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            discriminator.save(join(gan_path, "gan_discriminator_{0:04d}_{1:04d}.h5".format(gan_index, epoch)))
            gen_noise = np.random.normal(size=(batch_size, vec_size))
            gen_data_epoch = unnormalize_multivariate_data(generator.predict_on_batch(gen_noise), scaling_values)
            gen_da = xr.DataArray(gen_data_epoch.astype(out_dtype), coords={"p": np.arange(gen_data_epoch.shape[0]),
                                                                            "y": np.arange(gen_data_epoch.shape[1]),
                                                                            "x": np.arange(gen_data_epoch.shape[2]),
                                                                            "channel": np.arange(train_data.shape[-1])},
                                  dims=("p", "y", "x", "channel"),
                                  attrs={"long_name": "Synthetic data", "units": ""})
            gen_da.to_dataset(name="gen_patch").to_netcdf(join(gan_path,
                                                               "gan_gen_patches_{0:04d}_epoch_{1:04d}.nc".format(
                                                                   gan_index, epoch)),
                                                          encoding={"gen_patch": {"zlib": True,
                                                                                  "complevel": 1}})
            encoder.save(join(gan_path, "gan_encoder_{0:04d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
        if epoch == num_epochs[-1] and ind_encoder is not None:
            print("Training Independent Encoder {0:d}".format(gan_index))
            gen_vec = np.random.normal(size=(train_data.shape[0], vec_size))
            gen_data = generator.predict(gen_vec)
            ind_encoder.fit(gen_data, gen_vec, batch_size=batch_size, epochs=num_epochs[-1], verbose=2)
            ind_encoder.save(join(gan_path, "gan_indencoder_{0:04d}_{1:04d}.h5".format(gan_index, epoch)))
        time_history_index = pd.DatetimeIndex(time_history)
        history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history,
                                          gen_loss_history, gen_enc_loss_history]),
                               index=time_history_index, columns=hist_cols)
        history.to_csv(join(gan_path, "gan_loss_history_{0:04d}.csv".format(gan_index)), index_label="Time")
    time_history_index = pd.DatetimeIndex(time_history)
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history,
                                      gen_loss_history, 
                                      gen_enc_loss_history]),
                           index=time_history_index, columns=hist_cols)
    history.to_csv(join(gan_path, "gan_loss_history_{0:04d}.csv".format(gan_index)), index_label="Time")


def train_full_gan(train_data, generator, encoder, discriminator, combined_model, vec_size, gan_path, gan_index,
                   metrics=("accuracy", ),
                   batch_size=128, num_epochs=(1, 5, 10), scaling_values=None,
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
    batch_labels = np.zeros(batch_size, dtype=np.int32)
    batch_labels[:batch_half] = 1
    batch_vec = np.zeros((batch_size, vec_size), dtype=np.float32)
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]), dtype=np.float32)
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
            gen_data_epoch = unscale_multivariate_data(generator.predict_on_batch(gen_noise), scaling_values)
            gen_da = xr.DataArray(gen_data_epoch.astype(out_dtype), coords={"p": np.arange(gen_data_epoch.shape[0]),
                                                          "y": np.arange(gen_data_epoch.shape[1]),
                                                          "x": np.arange(gen_data_epoch.shape[2]),
                                                          "channel": np.arange(train_data.shape[-1])},
                                  dims=("p", "y", "x", "channel"),
                                  attrs={"long_name": "Synthetic data", "units": ""})
            gen_da.to_dataset(name="gen_patch").to_netcdf(join(gan_path,
                                                               "gan_gen_patches_{0:04d}_epoch_{1:03d}.nc".format(gan_index, epoch)),
                                                          encoding={"gen_patch": {"zlib": True,
                                                                                  "complevel": 1}})
            encoder.save(join(gan_path, "gan_encoder_{0:06d}_epoch_{1:04d}.h5".format(gan_index, epoch)))
            time_history_index = pd.DatetimeIndex(time_history)
            history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history, combined_loss_history]),
                                   index=time_history_index, columns=hist_cols)
            history.to_csv(join(gan_path, "gan_loss_history_{0:04d}.csv".format(gan_index)), index_label="Time")
    time_history_index = pd.DatetimeIndex(time_history)
    history = pd.DataFrame(np.hstack([current_epoch, disc_loss_history, combined_loss_history]),
                           index=time_history_index, columns=hist_cols)
    history.to_csv(join(gan_path, "gan_loss_history_{0:04d}.csv".format(gan_index)), index_label="Time")
    return


def train_gan(train_data, generator, discriminator, gan_path, gan_index, batch_size=128, num_epochs=(1, 5, 10),
              gen_optimizer="adam", disc_optimizer="adam", gen_input_size=100,
              gen_loss="binary_crossentropy", disc_loss="binary_crossentropy", metrics=("accuracy", ),
              encoder=None, encoder_loss="binary_crossentropy", scaling_values=None,
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
    combo_data_batch = np.zeros(np.concatenate([[batch_size], train_data.shape[1:]]), dtype=np.float32)
    batch_labels = np.zeros(batch_size, dtype=np.int32)
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
            gen_data_epoch = unscale_multivariate_data(generator.predict_on_batch(gen_noise), scaling_values)
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


def rescale_data(data, min_val, max_val):
    """
    Rescale data from -1 to 1 based on specified minimum and maximum values.

    Args:
        data: array of data
        min_val: minimum value
        max_val: maximum values

    Returns:

    """
    scaled_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
    return scaled_data


def rescale_multivariate_data(data, scaling_values=None):
    """
    Converts raw data into normalized values for each channel and then rescales the values from -1 to 1.

    Args:
        data: ndarray of shape (examples, y, x, variable)

    Returns:
        Normalized data, values used for scaling
    """
    normed_data = np.zeros(data.shape[:-1], dtype=np.float32)
    scaled_data = np.zeros(data.shape, dtype=np.float32)
    scale_cols = ["mean", "std", "min", "max", "max_mag"]
    set_scale = False
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
        set_scale = True
    for i in range(data.shape[-1]):
        if set_scale:
            scaling_values.loc[i, ["mean", "std"]] = [data[:, :, :, i].mean(), data[:, :, :, i].std()]
        normed_data[:] = (data[:, :, :, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
        if set_scale:
            scaling_values.loc[i, ["min", "max"]] = [normed_data.min(), normed_data.max()]
            scaling_values.loc[i, "max_mag"] = np.max(np.abs(scaling_values.loc[i, ["min", "max"]]))
        scaled_data[:, :, :, i] = rescale_data(normed_data,
                                               -scaling_values.loc[i, "max_mag"],
                                               scaling_values.loc[i, "max_mag"])
    return scaled_data, scaling_values


def unscale_data(data, min_val=0, max_val=255):
    """
    Scale data ranging from -1 to 1 back to its original range.

    Args:
        data:
        min_val:
        max_val:

    Returns:

    """
    unscaled_data = (data + 1) / 2 * (max_val - min_val) + min_val
    return unscaled_data


def unscale_multivariate_data(data, scaling_values):
    """
    Scale data ranging from -1 to 1 back to its original values.

    Args:
        data:
        scaling_values:

    Returns:

    """
    unscaled_data = np.zeros(data.shape[:-1], dtype=np.float32)
    unnormed_data = np.zeros(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        unscaled_data[:] = unscale_data(data[:, :, :, i],
                                        -scaling_values.loc[i, "max_mag"],
                                        scaling_values.loc[i, "max_mag"])
        unnormed_data[:, :, :, i] = unscaled_data * scaling_values.loc[i, "std"] + scaling_values.loc[i, "mean"]
    return unnormed_data


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
