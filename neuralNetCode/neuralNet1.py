from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Flatten, Reshape

sequenceLength = 1000
size = (1000 - 3) * 8

model = Sequential()

model.add(Convolution1D(8, 4, 'uniform', 'linear', border_mode='valid', 
                        subsample_length=1, input_dim=1, input_length=sequenceLength))
model.add(Flatten())

model.add(Dense(output_dim=size, init='uniform', activation='relu'))
model.add(Dense(128, 'uniform', 'relu'))
model.add(Dense(size, 'uniform', 'relu'))
model.add(Reshape(dims=(size, 1)))
model.add(Convolution1D(1, 4, 'uniform', 'linear', border_mode='valid', subsample_length=1, input_dim=1, input_length=size))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

#model.train()