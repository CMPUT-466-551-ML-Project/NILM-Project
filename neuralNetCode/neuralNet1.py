from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Flatten, Reshape

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
        #model.fit(X, Y, batch_size=10, nb_epoch=1)
        return

neuralNet = DenoisingAutoencoder(1000)  #compile example neural net
