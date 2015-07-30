#!/usr/bin/env python
""" Script to start development test battery for the grmpy package. We draw
random test cases for a fixed amount of time. Most tests itself are
distributed as part of the grmpy package in grmpy/tests. However, additional
tests are found in the modules subdirectory.
"""

# standard library
from datetime import timedelta
from datetime import datetime

import numpy as np

import argparse
import logging
import random
import sys
import os

# testing library
import modules.auxiliary as aux
import modules.battery as development_tests

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy.tests.test import Tests as package_tests

# virtual environment
if not hasattr(sys, 'real_prefix'):
    raise AssertionError('Please use a virtual environment for testing')

# module-wide variables
PYTHON_VERSION = sys.version_info[0]

''' Main Function.
'''
def run(hours):
    """ Run test battery.
    """

    # Generate time objects
    start, timeout = datetime.now(), timedelta(hours=hours)

    # Define list of admissible tests
    test_labels = ['0']
    package_labels = ['1', '2', '3', '4', '5']
    labels = test_labels + package_labels

    # Initialize counter
    dict_ = dict()

    for label in labels:

        dict_[label] = {}

        dict_[label]['success'] = 0

        dict_[label]['failure'] = 0

    # Logging.
    logger = logging.getLogger('DEV-TEST')

    msg = 'Initialization of a ' + str(hours) + \
          ' hours testing run with Python ' + \
          str(PYTHON_VERSION) + '.'

    logger.info(msg)

    # Evaluation loop
    while True:

        # Set seed
        seed = int(datetime.now().microsecond)

        np.random.seed(seed)

        # Setup of test case
        label = np.random.choice(labels)

        if label in test_labels:
            test = getattr(development_tests, 'test_' + label)
        elif label in package_labels:
            test = getattr(package_tests, 'test_' + label)
        else:
            raise AssertionError

        # Execute test
        try:

            test()

            dict_[label]['success'] += 1

        except Exception:

            dict_[label]['failure'] += 1

            msg = 'Failure for test ' + label + ' with seed ' + str(seed)

            logger.info(msg)

        # Check for timeout
        current = datetime.now()

        duration = current - start

        if timeout < duration:
            break

    # Finishing.
    return dict_

''' Execution of module as script.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run development test battery of grmToolbox.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--hours', action='store', dest='hours', type=float, default=1.0,
                        help='run time in hours')

    parser.add_argument('--notification', action='store_true', dest='notification', default=False, \
                        help='send notification')

    hours, notification = aux.distribute_input(parser)

    aux.start_logging()

    dict_ = run(hours)

    aux.finish(dict_, hours, notification)

    aux.cleanup()