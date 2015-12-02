#!/usr/bin/python2

import argparse
import logging
import os
import sys

from nilm.timeseries import TimeSeries


def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.DEBUG)
    log = logging.getLogger(__name__)

    device_files = [os.path.abspath(os.path.join(args.dir, p)) for p in
                    os.listdir(args.dir)]
    for dev_path in args.devices:
        try:
            device_files.remove(os.path.abspath(dev_path))
            dev_data = TimeSeries(os.path.basename(dev_path).split('.')[0],
                                  path=os.path.abspath(dev_path))
            dev_data.pad()
            dev_data.write(os.path.join(args.out, dev_data.name + '.dat'))
        except ValueError:
            log.error('Unable to find device file %s under %s' % (dev_path,
                                                                  args.dir))

    for agg_path in args.aggregated:
        try:
            device_files.remove(os.path.abspath(agg_path))
        except ValueError:
            log.warning('Unable to find aggregated power file %s under %s' %
                        (agg_path, args.dir))

    agg_data = TimeSeries(path=os.path.abspath(args.aggregated[0]))
    agg_data.pad()
    for agg_path in args.aggregated[1:]:
        agg_in = TimeSeries(path=os.path.abspath(agg_path))
        agg_in.pad()
        agg_data += agg_in

    for dev_path in device_files:
        dev_data = TimeSeries(path=dev_path)
        dev_data.pad()

        agg_data -= dev_data

    agg_data.write(os.path.join(args.out, 'aggregate.dat'))

    return 0


def get_parser():
    parser = argparse.ArgumentParser(description='Create a new aggregated power'
                                     ' file.')
    parser.add_argument('-o', '--out', required=True, help='Output directory.')
    parser.add_argument('-a', '--aggregated', nargs='+', required=True,
                        help='Aggregated power usage file.')
    parser.add_argument('-d', '--dir', required=True,
                        help='Directory containing device files.')
    parser.add_argument('--devices', nargs='+', required=True,
                        help='List of device files to keep.')
    parser.add_argument('-l', '--log', default='/tmp/agg.log',
                        help='File to write log to.')
    return parser


if __name__ == '__main__':
    sys.exit(main())
