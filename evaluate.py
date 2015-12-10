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
    
args = parser.parse_args()

channel = args.channel

result_file = '%s/%s' % (args.model, channel)
model_file = os.path.abspath(result_file + '.yml')
weight_file = os.path.abspath(result_file + '.h5')
data_file = os.path.abspath(result_file + '.dat')

device_file = '%s/%s.dat' % (args.dir, channel)
agg_file = args.aggregated

aggregate = TimeSeries(path=os.path.abspath(agg_file))
aggregate.array = aggregate.array[len(aggregate.array)/5*4+1:]
true_device = TimeSeries(path=os.path.abspath(device_file))
true_device.intersect(aggregate)


activations = true_device.activations(np.float32(25.0))

if len(activations) == 0:
    print 'No activations found.'
    exit(0)

with open(data_file, 'r') as data:
    (window_size, std_dev, max_energy) = [x for x in data.readline().split()]
    window_size = int(window_size)
    std_dev = float(std_dev)
    max_energy = float(max_energy)

print 'Window size: %s' % window_size
print 'Std dev: %s' % std_dev
print 'Max power: %s' % max_energy

network = DenoisingAutoencoder(window_size, model_path=model_file,
                               weight_path=weight_file)

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

dev_windows = dev_windows * max_energy

windows_with_activations = []
truth_activations = []
for (i, window) in enumerate(dev_windows):
    if not any([e > np.float(25.0) for e in truth_windows[i]]):
        continue

    windows_with_activations.append(window)
    truth_activations.append(truth_windows[i])
    
    print 'Mean: %s' % window.mean(axis=None)
    #t1 = TimeSeries()
    #t1.powers = window
    #t2 = TimeSeries()
    #t2.powers = true_windows[i]
    print window
    print truth_windows[i]
    print 'Error: %s' % root_mean_squared_error(window, truth_windows[i])
    
    #print f_score(t1, t2)
    
def CalculateFrom2DArrays(dev_windows, completeTruth):
    length = dev_windows.shape[0] * dev_windows.shape[1]
    t1 = TimeSeries()
    allWindows = dev_windows.reshape(length)
    t1.array.resize(len(allWindows))
    t1.powers = allWindows

    t2 = TimeSeries()
    allTruth = completeTruth.reshape(length)
    t2.array.resize(len(allWindows))
    t2.powers = allTruth

    print 'Over %s time intervals:' % length
    print 'Precision: %s Error: %s' % (f_score(t1, t2, 20.0), root_mean_squared_error(allWindows, allTruth))

completeTruth = np.array(truth_windows)

print '\n\nOver all timesteps:'
print 'Max power: %s' % max_energy
CalculateFrom2DArrays(dev_windows, completeTruth)


print '\n\nOver windows with some activations:'
print 'Activations:', len(activations)
CalculateFrom2DArrays(np.array(windows_with_activations), np.array(truth_activations))