import os

import numpy as np

from nilm.evaluation import f_score, root_mean_squared_error
from nilm.network import DenoisingAutoencoder
from nilm.timeseries import TimeSeries


channel = 'channel_6'
result_file = '../results/house1/edge/%s' % channel
model_file = os.path.abspath(result_file + '.yml')
weight_file = os.path.abspath(result_file + '.h5')

device_file = '../out-data/house1/%s.dat' % channel
agg_file = '../out-data/house1/aggregate.dat'

max_energy = np.float32(1422.0)

aggregate = TimeSeries(path=os.path.abspath(agg_file))
aggregate.array = aggregate.array[len(aggregate.array)/5*4+1:]
true_device = TimeSeries(path=os.path.abspath(device_file))
true_device.intersect(aggregate)

window_size = 307
network = DenoisingAutoencoder(window_size, model_path=model_file,
                               weight_path=weight_file)

std_dev = np.float32(148.485)
truth_windows = []
agg_windows = []

for i in xrange(0, len(aggregate.array) - window_size + 1, window_size):
    truth_windows.append(true_device.powers[i:i+window_size-6])
    agg_window = aggregate.powers[i:i+window_size]

    mean = agg_window.mean(axis=None)
    agg_window = np.divide(np.subtract(agg_window, mean), std_dev)
    agg_windows.append(agg_window)

agg_windows = np.array(agg_windows)
agg_windows.shape = (len(agg_windows), window_size, 1)

dev_windows = network.model.predict(agg_windows)
dev_windows.shape = (len(agg_windows), window_size - 6)
print dev_windows.shape


for (i, window) in enumerate(dev_windows):
    if not any([e > np.float(25.0) for e in truth_windows[i]]):
        continue

    window = window * max_energy

    print 'Mean: %s' % window.mean(axis=None)
    #t1 = TimeSeries()
    #t1.powers = window
    #t2 = TimeSeries()
    #t2.powers = true_windows[i]
    print window
    print truth_windows[i]
    print 'Error: %s' % root_mean_squared_error(window, truth_windows[i])
    #print f_score(t1, t2)
