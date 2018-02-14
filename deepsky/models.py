from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD
from keras.models import Model, save_model, load_model
import pickle
import inspect
from os.path import join
import yaml
from keras.regularizers import l2
from deepsky.gan import generator_model, encoder_model, discriminator_model, stack_gen_disc, stack_gen_enc, train_gan_quiet
import keras.backend as K
import numpy as np


class LogisticPCA(BaseEstimator):
    def __init__(self, n_components=5, penalty="l1", C=0.1):
        self.n_components = n_components
        self.C = C
        self.penalty = penalty
        self.pca = []
        self.model = LogisticRegression(penalty=penalty, C=self.C)

    def fit(self, X, y):
        X_pca = np.zeros((X.shape[0], self.n_components * X.shape[2]))
        c = 0
        for i in range(X.shape[2]):
            self.pca.append(PCA(n_components=self.n_components))
            X_pca[:, c:c + self.n_components] = self.pca[-1].fit_transform(X[:, :, i])
            c += self.n_components
        self.model.fit(X_pca, y)

    def transform(self, X):
        X_pca = np.zeros((X.shape[0], self.n_components * X.shape[2]))
        c = 0
        for i in range(X.shape[2]):
            X_pca[:, c:c + self.n_components] = self.pca[i].transform(X[:, :, i])
            c += self.n_components
        return X_pca

    def predict(self, X):
        X_pca = self.transform(X)
        return self.model.predict(X_pca)

    def predict_proba(self, X):
        X_pca = self.transform(X)
        return self.model.predict_proba(X_pca)


class LogisticGAN(BaseEstimator):
    def __init__(self, data_width=32, num_input_channels=15, filter_width=5, min_conv_filters=16,
                 min_data_width=4, encoding_channels=100, activation="relu",
                 dropout_alpha=0, output_activation="linear", stride=2, num_epochs=10,
                 batch_size=128, learning_rate=0.001, beta_one=0.5, index=0, penalty="l1", C=0.01):
        self.data_width = data_width
        self.index = index
        self.num_input_channels = num_input_channels
        self.filter_width = filter_width
        self.min_conv_filters = min_conv_filters
        self.min_data_width = min_data_width
        self.encoding_channels = encoding_channels
        self.activation = activation
        self.dropout_alpha = dropout_alpha
        self.output_activation = output_activation
        self.stride = stride
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_one = beta_one
        self.penalty = penalty
        self.C = C
        self.gen, self.gen_input = generator_model(input_size=self.encoding_channels,
                                                     filter_width=self.filter_width,
                                                     min_data_width=self.min_data_width,
                                                     min_conv_filters=min_conv_filters,
                                                     output_size=(self.data_width, self.data_width,
                                                                  self.num_input_channels),
                                                     stride=self.stride,
                                                     activation=self.activation,
                                                     output_activation=self.output_activation,
                                                     dropout_alpha=self.dropout_alpha)
        self.disc, self.disc_input = discriminator_model(input_size=(self.data_width,
                                                               self.data_width,
                                                               self.num_input_channels),
                                                   filter_width=self.filter_width,
                                                   min_data_width=self.min_data_width,
                                                   min_conv_filters=self.min_conv_filters,
                                                   stride=self.stride,
                                                   activation=self.activation,
                                                   dropout_alpha=self.dropout_alpha)
        self.enc, self.enc_input = encoder_model(input_size=(self.data_width,
                                                   self.data_width,
                                                   self.num_input_channels),
                                                   filter_width=self.filter_width,
                                                   min_data_width=self.min_data_width,
                                                   min_conv_filters=self.min_conv_filters,
                                                   output_size=self.encoding_channels,
                                                   stride=self.stride,
                                                   activation=self.activation,
                                                   dropout_alpha=self.dropout_alpha)

        optimizer = Adam(lr=self.learning_rate, beta_1=self.beta_one, clipnorm=1.)
        self.discriminator = Model(self.disc_input, self.disc)
        self.discriminator.compile(optimizer=optimizer,
                                   loss="binary_crossentropy")
        self.generator = Model(self.gen_input, self.gen)
        self.generator.compile(optimizer=optimizer,
                               loss="mse")
        self.gen_disc = stack_gen_disc(self.generator, self.discriminator)
        self.gen_disc.compile(optimizer=optimizer,
                              loss="binary_crossentropy")
        self.encoder = Model(self.enc_input, self.enc)
        self.encoder.compile(optimizer=optimizer,
                                   loss="mse")
        self.gen_enc = stack_gen_enc(self.generator, self.encoder)
        self.gen_enc.compile(optimizer=optimizer,
                             loss="mse")
        print("Generator")
        print(self.generator.summary())
        print("Discriminator")
        print(self.discriminator.summary())
        print("Encoder")
        print(self.encoder.summary())
        print("Gen Disc")
        print(self.gen_disc.summary())
        print("Gen Enc")
        print(self.gen_enc.summary())
        self.logistic = LogisticRegression(penalty=self.penalty, C=self.C, solver="saga", verbose=1)
        return

    def fit(self, X, y):
        train_gan_quiet(X, self.generator, self.discriminator, self.gen_disc,
                        self.gen_enc, self.encoding_channels, self.batch_size,
                        self.num_epochs, self.index)
        print("Transform X" + str(self.index))
        encoded_X = self.transform(X)
        print("Encoded X " + str(self.index), encoded_X.shape)
        print("Train Logistic " + str(self.index))
        self.logistic.fit(encoded_X, y)

    def transform(self, X):
        return self.encoder.predict(X)

    def predict(self, X):
        encoded_X = self.transform(X)
        return self.logistic.predict(encoded_X)

    def predict_proba(self, X):
        encoded_X = self.transform(X)
        return self.logistic.predict_proba(encoded_X)


def save_logistic_gan(log_gan_model, out_path):
    save_model(log_gan_model.generator, 
               join(out_path, "logistic_gan_{0}_generator.h5".format(log_gan_model.index))) 
    save_model(log_gan_model.discriminator, 
               join(out_path, "logistic_gan_{0}_discriminator.h5".format(log_gan_model.index))) 
    save_model(log_gan_model.encoder, 
               join(out_path, "logistic_gan_{0}_encoder.h5".format(log_gan_model.index))) 
    with open(join(out_path, "logistic_gan_{0}_logistic.pkl".format(log_gan_model.index)), "wb") as logistic_file:
        pickle.dump(log_gan_model.logistic, logistic_file, pickle.HIGHEST_PROTOCOL)

    model_args = inspect.signature(log_gan_model.__init__).args
    if "self" in model_args:
        model_args.remove("self")
    param_dict = {}
    for arg in model_args:
        if hasattr(log_gan_model, arg):
            param_dict[arg] = getattr(log_gan_model, arg)
    with open(join(out_path, "logistic_gan_{0}_params.yaml".format(log_gan_model.index)), "w") as yaml_file:
        yaml.dump(param_dict, 
                  yaml_file,
                  default_flow_style=False)
    return


def load_logistic_gan(path, index):
    with open(join(path, "logistic_gan_{0}_params.yaml".format(index))) as yaml_file:
        param_dict = yaml.load(yaml_file)
    gan_obj = LogisticGAN(**param_dict)
    gan_obj.discriminator.load_weights(join(path, "logistic_gan_{0}_discriminator.h5".format(index)))
    gan_obj.encoder.load_weights(join(path, "logistic_gan_{0}_encoder.h5".format(index)))
    gan_obj.generator.load_weights(join(path, "logistic_gan_{0}_generator.h5".format(index)))
    with open(join(path, "logistic_gan_{0}_logistic.pkl".format(index)), "rb") as logistic_file:
        gan_obj.logistic = pickle.load(logistic_file)
    return gan_obj


def hail_conv_net(data_width=32, num_input_channels=1, filter_width=5, min_conv_filters=16,
                  filter_growth_rate=2, min_data_width=4,
                  dropout_alpha=0, activation="relu", regularization_alpha=0.01, optimizer="sgd",
                  learning_rate=0.001, loss="mse", metrics=("mae", "auc"), **kwargs):
    device = "/gpu:0"
    with K.tf.device(device):
        cnn_input = Input(shape=(data_width, data_width, num_input_channels))
        num_conv_layers = int(np.log2(data_width) - np.log2(min_data_width))
        num_filters = min_conv_filters
        cnn_model = cnn_input
        for c in range(num_conv_layers):
            cnn_model = Conv2D(num_filters, (filter_width, filter_width), strides=2, padding="same",
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
    cnn_model_complete.compile(optimizer=opt, loss=loss, metrics=metrics)
    return cnn_model_complete
