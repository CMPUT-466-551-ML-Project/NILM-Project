#!/usr/bin/python2

import argparse
import logging
import os
import sys

import numpy as np

from nilm.network import DenoisingAutoencoder
from nilm.preprocess import (confidence_estimator, get_changed_data,
                             solve_constant_energy, sort_data)
from nilm.timeseries import TimeSeries


log = logging.getLogger(__name__)


def apply_preprocess(aggregated, devices, method, threshold=np.float32(0.0)):
    """
    Apply the given preprocessing method to the devices.
    """
    if method == 'raw':
        return

    indicators = [d.indicators(threshold) for d in devices]

    if method == 'constant':
        (energies, _) = solve_constant_energy(aggregated, indicators)

        for (e, d) in zip(energies, devices):
            log.info('Setting constant energy %s for device %s.' % (e, d.name))
            d.powers = e * d.indicators(np.float32(10))

    elif method == 'interval':
        energy_dict = confidence_estimator(aggregated, devices, sort_data,
                                           threshold)

        for d in devices:
            log.info('Setting constant energy %s for device %s.' %
                     (energy_dict[d.name], d.name))
            d.powers = energy_dict[d.name] * d.indicators(np.float32(10))

    elif method == 'edge':
        energy_dict = confidence_estimator(aggregated, devices,
                                           get_changed_data, threshold)

        for d in devices:
            log.info('Setting constant energy %s for device %s.' %
                     (energy_dict[d.name], d.name))
            d.powers = energy_dict[d.name] * d.indicators(np.float32(10))


def main():
    parser = get_parser()
    args = parser.parse_args()
    format = '%(asctime)s %(message)s'
    logging.basicConfig(filename=args.log, level=logging.DEBUG, format=format)

    device_files = [os.path.abspath(os.path.join(args.dir, p)) for p in
                    os.listdir(args.dir)]
    agg_path = os.path.abspath(args.aggregated)
    if agg_path in device_files:
        device_files.remove(agg_path)

    agg_data = TimeSeries(path=agg_path)
    agg_data.array = agg_data.array[0:len(agg_data.array)/5*4]

    device_in = []
    for dev_path in device_files:
        dev = TimeSeries(os.path.basename(dev_path).split('.')[0],
                         path=dev_path)
        dev.intersect(agg_data)
        device_in.append(dev)

    apply_preprocess(agg_data.powers, device_in, args.preprocess,
                     np.float32(25.00))

    for dev in device_in:
        log.info('Training: %s' % dev.name)
        activations = dev.activations(np.float32(25.0))

        log.info('Activations:')
        for a in activations:
            log.info('From %s to %s lasting %s' % (a[0], a[1], a[1] - a[0]))
        if len(activations) == 0:
            log.info('No activations found.')
            continue

        window_size = sum([a[1] - a[0] for a in activations])/len(activations)
        length = min(len(dev.array), len(agg_data.array))

        log.info('Window size: %s' % window_size)
        log.info('Series length: %s' % length)

        input_windows = []
        label_windows = []

        log.info('Computing windows...')
        std_dev = np.std(np.random.choice(dev.powers, 10000))
        max_power = dev.powers.max()

        log.info('Std Dev: %s' % std_dev)
        log.info('Max Power: %s' % max_power)
        for i in xrange(0, length - window_size + 1, window_size):
            label_window = dev.powers[i:i+window_size]
            input_window = np.divide(agg_data.powers[i:i+window_size],
                                     max_power)

            mean = input_window.mean(axis=None)
            label_window = np.divide(np.subtract(input_window, mean), std_dev)

            label_windows.append(input_window)
            input_windows.append(label_window)

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
    parser.add_argument('-p', '--preprocess',
                        choices=['raw','constant','interval','edge'],
                        default='raw',
                        help='Which preprocessing algorithm to use.')
    return parser


if __name__ == '__main__':
    sys.exit(main())
