#!/usr/bin/python2

import argparse
import logging
import os
import sys

import numpy as np

from nilm.network import DenoisingAutoencoder
from nilm.timeseries import TimeSeries


def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.DEBUG)
    log = logging.getLogger(__name__)

    device_files = [os.path.abspath(os.path.join(args.dir, p)) for p in
                    os.listdir(args.dir)]
    agg_path = os.path.abspath(args.aggregated)
    if agg_path in device_files:
        device_files.remove(agg_path)

    agg_data = TimeSeries(path=agg_path)

    device_in = []
    for dev_path in device_files:
        dev = TimeSeries(os.path.basename(dev_path).split('.')[0],
                         path=dev_path)
        dev.intersect(agg_data)
        device_in.append(dev)

    for dev in device_in:
        log.info('Training: %s' % dev.name)
        activations = dev.activations(np.float32(10.0))
        window_size = sum([a[1] - a[0] for a in activations])/len(activations)
        length = min(len(dev.array), len(agg_data.array))
        log.info('Window size: %s' % window_size)
        log.info('Series length: %s' % length)

        input_windows = []
        label_windows = []

        log.info('Computing windows...')
        std_dev = np.std(np.random.choice(dev.powers, 10000))
        max_power = dev.powers.max()
        for i in xrange(0, length - window_size + 1, window_size):
            input_window = dev.powers[i:i+window_size]
            label_window = np.divide(agg_data.powers[i:i+window_size],
                                     max_power)

            mean = input_window.mean(axis=None)
            input_window = np.divide(np.subtract(input_window, mean), std_dev)

            input_windows.append(input_window)
            label_windows.append(label_window)

        input_windows = np.array(input_windows)
        label_windows = np.array(label_windows)
        input_windows.shape = (len(input_windows), window_size, 1)
        label_windows.shape = (len(label_windows), window_size, 1)

        log.info('Training network...')
        network = DenoisingAutoencoder(window_size)
        network.train(input_windows, label_windows)

        log.info('Saving model to: %s' % os.path.join(args.out,
                                                      dev.name + '.yml'))
        network.save_model(os.path.join(args.out, dev.name + '.yml'))

        log.info('Saving weight to: %s' % os.path.join(args.out,
                                                       dev.name + '.h5'))
        network.save_weights(os.path.join(args.out, dev.name + '.h5'))

    return 0


def get_parser():
    parser = argparse.ArgumentParser(description='Train neural networks for the'
                                     ' given devices.')
    parser.add_argument('-o', '--out', required=True, help='Output directory.')
    parser.add_argument('-a', '--aggregated', required=True,
                        help='Aggregated power usage file.')
    parser.add_argument('-d', '--dir', required=True,
                        help='Directory containing device files.')
    parser.add_argument('-l', '--log', default='/tmp/agg.log',
                        help='File to write log to.')
    return parser


if __name__ == '__main__':
    sys.exit(main())
