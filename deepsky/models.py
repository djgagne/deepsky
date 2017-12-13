from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD
from keras.models import Model
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