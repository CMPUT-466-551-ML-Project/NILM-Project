import os

import argparse
import numpy as np

from nilm.evaluation import f_score, root_mean_squared_error, mean_squared_error
from nilm.network import DenoisingAutoencoder
from nilm.timeseries import TimeSeries

# Set up arg parser
parser = argparse.ArgumentParser(description='Run evaluation on a previously generated neural network.')
parser.add_argument('-a', '--aggregated', required=True,
                    help='Aggregated power usage file.')
                    
# Need results folder, and aggregated data folder
parser.add_argument('-d', '--dir', required=True,
                    help='Directory containing device files.')
parser.add_argument('-m', '--model', required=True,
                    help='Directory containing neural network model files.')
parser.add_argument('-c', '--channel', required=True, help='Device channel')
parser.add_argument('-l', '--log', default='/tmp/agg.log',
                    help='File to write log to.')
parser.add_argument('-p', '--preprocess',
                    choices=['raw','constant','interval','edge'],
                    default='raw',
                    help='Which preprocessing algorithm to use.')
    
args = parser.parse_args()

channel = args.channel

result_file = '%s/%s' % (args.model, channel)
model_file = os.path.abspath(result_file + '.yml')
weight_file = os.path.abspath(result_file + '.h5')

device_file = '%s/%s.dat' % (args.dir, channel)
agg_file = args.aggregated

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

totalError = 0
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
    totalError += mean_squared_error(window, truth_windows[i])
    #print f_score(t1, t2)

print 'Error: %s %s' % (totalError, np.sqrt(totalError))