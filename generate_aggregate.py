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

    return 0


def get_parser():
    parser = argparse.ArgumentParser(description='Create a new aggregated power'
                                     ' file.')
    parser.add_argument('-o', '--out', required=True, help='Output file.')
    parser.add_argument('-a', '--aggregated', required=True,
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
