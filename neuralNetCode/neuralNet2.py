from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Flatten, Reshape
import numpy as np

import pandas as pd


class DenoisingAutoencoder(object):
    
    def __init__(self, windowSize):
        self.windowSize = windowSize
        num_filters = 8
        self.size = (windowSize - 3) * num_filters
        self.model = Sequential()
        
        self.model.add(Convolution1D(num_filters, 4, 'uniform', 'linear', border_mode='valid', 
                                     subsample_length=1, input_dim=1, input_length=windowSize))
        self.model.add(Flatten())
        
        self.model.add(Dense(output_dim=self.size, init='uniform', activation='relu'))
        self.model.add(Dense(128, 'uniform', 'relu'))
        self.model.add(Dense(self.size, 'uniform', 'relu'))
        self.model.add(Reshape(dims=(self.windowSize - 3, num_filters)))
        self.model.add(Convolution1D(1, 4, 'uniform', 'linear', border_mode='valid',
                                     subsample_length=1, input_dim=1, input_length=self.size))
        
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')        
        
    def train(self, X, Y):
        model.fit(X, Y, batch_size=10, nb_epoch=1)
        return
    
class Device(object):
    
    def __init__(self, number, windowSize, maxPowerDemand):
        self.number = number
        self.windowSize = windowSize
        self.maxPowerDemand = maxPowerDemand

numDevices = 4
selectedDevices = [Device('5', 1000, 35), Device('11', 1000, 35), 
                   Device('13', 1000, 35), Device('14', 1000, 35)]
neuralNets = []

consumption = pd.read_csv('chunk_firsttwelfhours_consumption.csv', index_col=0, header=0)
x = consumption['1'].as_matrix()

std = np.std(np.random.choice(x, 10000))

for device in selectedDevices:
    y = consumption[device.number].as_matrix()
    length = min(len(x), len(y))
    
    xWindows = []
    yWindows = []
    stepSize = 10000
    
    for j in range(0, length - device.windowSize + 1, stepSize):
        xWindow = x[j:j+device.windowSize]
        yWindow = y[j + 3:j+device.windowSize - 3]
        # Standardize the input and target
        average = np.average(xWindow)
        for i in range(device.windowSize):
            if i < device.windowSize - 6:
                yWindow[i] = yWindow[i] / device.maxPowerDemand
            xWindow[i] = xWindow[i] - average
            xWindow[i] = xWindow[i] / std
        
        xWindows.append(xWindow)
        yWindows.append(yWindow)
        
    xWindows = np.array(xWindows)
    yWindows = np.array(yWindows)
    xWindows.shape = (len(xWindows), device.windowSize, 1)
    yWindows.shape = (len(yWindows), device.windowSize - 6, 1)
    
    neuralNet = DenoisingAutoencoder(device.windowSize)
    neuralNet.train(xWindows, yWindows)
    neuralNets.append(neuralNet)