#!/usr/bin/env python
""" Script to start development test battery for the grmpy package. We draw
random test cases for a fixed amount of time. The tests are defined in tests.py
, which if found in the modules subdirectory.
"""

# standard library
from datetime import timedelta
from datetime import datetime

import numpy as np

import argparse
import logging
import random
import sys

# project library
from modules.auxiliary import *
import modules.tests as lib

# virtual environment
if not hasattr(sys, 'real_prefix'):
   raise AssertionError('Please use a virtual environment for testing')

PYTHON_VERSION = sys.version_info[0]

''' Main Function.
'''
def run(hours):
    """ Run test battery.
    """

    # Generate time objects
    start, timeout = datetime.now(), timedelta(hours=hours)

    # Define list of admissible tests
    labels = ['A', 'B', 'C', 'D', 'E']

    # Initialize counter
    dict_ = dict()

    for label in labels:

        dict_[label] = {}

        dict_[label]['success'] = 0

        dict_[label]['failure'] = 0

    # Logging.
    logger = logging.getLogger('DEV-TEST')

    msg = 'Initialization of a ' + str(hours) + ' hours testing run with Python ' + str(PYTHON_VERSION) + '.'

    logger.info(msg)

    # Evaluation loop
    while True:

        # Set seed
        seed = int(datetime.now().microsecond)

        np.random.seed(seed)

        # Setup of test case
        label = np.random.choice(labels)

        test = getattr(lib,'test_' + label)

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

    hours, notification = distribute_input(parser)

    start_logging()

    dict_ = run(hours)

    finish(dict_, hours, notification)

    cleanup()