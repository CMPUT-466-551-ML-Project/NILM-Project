from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Flatten, Reshape
import numpy as np

import pandas as pd


class DenoisingAutoencoder(object):
    
    def __init__(self, windowSize):
        self.windowSize = windowSize
        self.size = (windowSize - 3) * 8
        self.model = Sequential()
        
        self.model.add(Convolution1D(8, 4, 'uniform', 'linear', border_mode='valid', 
                                     subsample_length=1, input_dim=1, input_length=windowSize))
        self.model.add(Flatten())
        
        self.model.add(Dense(output_dim=self.size, init='uniform', activation='relu'))
        self.model.add(Dense(128, 'uniform', 'relu'))
        self.model.add(Dense(self.size, 'uniform', 'relu'))
        self.model.add(Reshape(dims=(self.size, 1)))
        self.model.add(Convolution1D(1, 4, 'uniform', 'linear', border_mode='valid',
                                     subsample_length=1, input_dim=1, input_length=self.size))
        
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')        
        
    def train(self, X, Y):
        model.fit(X, Y, batch_size=10, nb_epoch=1)
        return

numDevices = 4
selectedDevices = ['5', '11', '13', '14']
windowSize = [1000, 1000, 1000, 1000]
neuralNets = []

consumption = pd.read_csv('chunk_firsttwelfhours_consumption.csv', index_col=0, header=0)
x = consumption['1']

for i in range(numDevices):
    
    y = consumption[selectedDevices[i]]
    length = min(len(x), len(y))
    print length
    xWindows = []
    yWindows = []
    stepSize = 10000
    for j in range(0, length - windowSize[i] + 1, stepSize):
        xWindows.append(np.array(x[j:j+windowSize[i]]))
        yWindows.append(np.array(y[j:j+windowSize[i]]))
        
    xWindows = np.array(xWindows)
    yWindows = np.array(yWindows)
    neuralNet = DenoisingAutoencoder(windowSize[i])
    neuralNet.train(xWindows, yWindows)
    neuralNets.append(neuralNet)