"""
Set of neural networks for use with NILM task.
"""

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Flatten, Reshape


class DenoisingAutoencoder(object):
    """
    Neural network which implements a denoising auto-encoder. Inputs are
    convoluted before being fed into an encoding layer. From the encoding layer
    we learn to recover the original signal.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.size = (window_size - 3) * 8
        self.initialize_model()

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
