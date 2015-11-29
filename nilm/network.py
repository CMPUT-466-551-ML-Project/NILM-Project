"""
Set of neural networks for use with NILM task.
"""

import os

from keras.models import Sequential, model_from_yaml
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Flatten, Reshape


class DenoisingAutoencoder(object):
    """
    Neural network which implements a denoising auto-encoder. Inputs are
    convoluted before being fed into an encoding layer. From the encoding layer
    we learn to recover the original signal.
    """
    def __init__(self, window_size, model_path=None, weight_path=None):
        self.window_size = window_size
        self.size = (window_size - 3) * 8

        if model_path is not None:
            self.load_model(model_path)
        else:
            self.initialize_model()

        if weight_path is not None:
            self.load_weights(weight_path)

    def initialize_model(self):
        """Initialize the network model."""
        self.model = Sequential()

        self.model.add(Convolution1D(8, 4, 'uniform', 'linear',
                                     border_mode='valid', subsample_length=1,
                                     input_dim=1,
                                     input_length=self.window_size))
        self.model.add(Flatten())
        self.model.add(Dense(output_dim=self.size, init='uniform',
                             activation='relu'))
        self.model.add(Dense(128, 'uniform', 'relu'))
        self.model.add(Dense(self.size, 'uniform', 'relu'))
        self.model.add(Reshape(dims=(self.size, 1)))
        self.model.add(Convolution1D(1, 4, 'uniform', 'linear',
                                     border_mode='valid', subsample_length=1,
                                     input_dim=1, input_length=self.size))

        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')

    def train(self, aggregate_power, device_power):
        """Train the network given the aggregate and device powers."""
        self.model.fit(aggregate_power, device_power, batch_size=10, nb_epoch=1)

    def save_model(self, path):
        """Save the network model to the given path as yaml."""
        yaml_string = self.model.to_yaml()
        path = os.path.abspath(path)

        with open(path, 'w') as fd:
            fd.write(yaml_string)

    def save_weights(self, path):
        """Save the network weights to the given path in HDF5."""
        path = os.path.abspath(path)
        self.model.save_weights(path, overwrite=True)

    def load_model(self, model_path):
        """ Load the network model from the given path."""
        model_path = os.path.abspath(model_path)
        with open(model_path, 'r') as fd:
            self.model = model_from_yaml(fd.read())

    def load_weights(self, weight_path):
        """Load the network weights from the given path."""
        weight_path = os.path.abspath(weight_path)
        self.model.load_weights(weight_path)
