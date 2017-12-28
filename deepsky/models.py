from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.regularizers import l2
from deepsky.gan import generator_model, encoder_disc_model, stack_gen_disc, stack_gen_enc, train_gan_quiet
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
                 batch_size=128, learning_rate=0.0001, beta_one=0.5, penalty="l1", C=0.01):
        self.data_width = data_width
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
        generator, generator_input = generator_model(input_size=self.encoding_channels,
                                                     filter_width=self.filter_width,
                                                     min_data_width=self.min_data_width,
                                                     min_conv_filters=min_conv_filters,
                                                     output_size=(self.data_width, self.data_width,
                                                                  self.num_input_channels),
                                                     stride=self.stride,
                                                     activation=self.activation,
                                                     output_activation=self.output_activation,
                                                     dropout_alpha=self.dropout_alpha)
        disc, enc, disc_input = encoder_disc_model(input_size=(self.data_width,
                                                               self.data_width,
                                                               self.num_input_channels),
                                                   filter_width=self.filter_width,
                                                   min_data_width=self.min_data_width,
                                                   min_conv_filters=self.min_conv_filters,
                                                   output_size=self.encoding_channels,
                                                   stride=self.stride,
                                                   activation=self.activation,
                                                   encoder_output_activation=self.output_activation,
                                                   dropout_alpha=self.dropout_alpha)
        self.generator = Model(generator_input, generator)
        self.generator.compile(optimizer=Adam(lr=self.learning_rate, beta_one=self.beta_one),
                               loss="binary_crossentropy")
        self.discriminator = Model(disc_input, disc)
        self.discriminator.compile(optimizer=Adam(lr=self.learning_rate, beta_one=self.beta_one),
                                   loss="binary_crossentropy")
        self.encoder = Model(disc_input, enc)
        self.encoder.compile(optimizer=Adam(lr=self.learning_rate, beta_one=self.beta_one),
                                   loss="mse")
        self.gen_disc = stack_gen_disc(self.generator, self.discriminator)
        self.gen_disc.compile(optimizer=Adam(lr=self.learning_rate, beta_one=self.beta_one),
                              loss="binary_crossentropy")
        self.gen_enc = stack_gen_enc(self.generator, self.encoder, self.discriminator)
        self.gen_enc.compile(optimizer=Adam(lr=self.learning_rate, beta_one=self.beta_one),
                             loss="mse")
        self.logistic = LogisticRegression(penalty=self.penalty, C=self.C)
        return

    def fit(self, X, y):
        train_gan_quiet(X, self.generator, self.discriminator, self.gen_disc,
                        self.gen_enc, self.encoding_channels, self.batch_size,
                        self.num_epochs, 0)
        encoded_X = self.transform(X)
        self.logistic.fit(encoded_X, y)

    def transform(self, X):
        return self.encoder.predict(X, batch_size=self.batch_size)

    def predict(self, X):
        encoded_X = self.transform(X)
        return self.logistic.predict(encoded_X)

    def predict_proba(self, X):
        encoded_X = self.transform(X)
        return self.logistic.predict_proba(encoded_X)


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
