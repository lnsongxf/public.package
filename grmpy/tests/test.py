#!/usr/bin/env python
""" Module for unit tests related to the parameter management and updating.
"""
# standard library
import signal
import shutil
import glob
import sys
import os

import numpy as np


# module variables
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.getcwd()

# testing library
from nose.core import runmodule
from nose.tools import assert_almost_equal

# Setting up signal handler
def signal_handler(signum, frame):
    raise TimedOutError

signal.signal(signal.SIGALRM, signal_handler)

# PYTHONPATH
dir_ = FILE_PATH.replace('/tests', '')
sys.path.insert(0, dir_)

# project library
import grmpy
import grmpy.tools.user as user
import grmpy.tests.randominit as aux
from grmpy.tests.exceptions import TimedOutError

# Module-wide variables
NUM_RUNS = 1
SMALL = 10e-10

''' Test class.
'''


class Tests(object):
    """ Test class.
    """

    @staticmethod
    def setup_class():
        """ Setup before any methods in this class.
        """
        os.chdir(FILE_PATH)

    @staticmethod
    def teardown_class():
        """ Teardown after any methods in this class.
        """
        os.chdir(TEST_PATH)

    def teardown(self):
        """ Teardown after each test method.
        """
        self.cleanup()

    @staticmethod
    def setup():
        """ Setup before each test method.
        """

    @staticmethod
    def cleanup():
        """ Cleanup after test battery.
        """
        files = []

        files = files + glob.glob('*.grmpy.*')

        files = files + glob.glob('*.ini')

        files = files + glob.glob('*.pkl')

        files = files + glob.glob('*.txt')

        files = files + glob.glob('*.dat')

        for file_ in files:

            if 'logging' in file_:
                continue

            try:

                os.remove(file_)

            except OSError:

                pass

            try:

                shutil.rmtree(file_)

            except OSError:

                pass

    @staticmethod
    def test_1():
        """ Test parameter transformations.
        """
        aux.generate_init_file()

        grmpy.simulate('test.grmpy.ini')

        init_dict = user.process_input('test.grmpy.ini')

        # Make sure to use simulation
        init_dict['DATA']['source'] = init_dict['SIMULATION']['target']
        init_dict['DATA']['agents'] = init_dict['SIMULATION']['agents']

        model_obj = user.construct_model(init_dict)

        paras_obj = user.construct_paras(init_dict, model_obj, False)

        para_objs = paras_obj.get_attr('para_objs')

        for para_obj in para_objs:
            value = para_obj.get_attr('value')

            ext = paras_obj._transform_to_external(para_obj, value)

            int_ = paras_obj._transform_to_internal(para_obj, ext)

            assert_almost_equal(value, int_)

    @staticmethod
    def test_2():
        """ Testing if a ten random initialization files can be used to simulate a
        model.
        """

        for _ in range(10):
            # Generate a random initialization file.
            aux.generate_init_file()

            # Simulation
            grmpy.simulate('test.grmpy.ini')

    @staticmethod
    def test_3():
        """ Testing if a random estimation task can be handled without complaints
        for five seconds.
        """

        for _ in range(NUM_RUNS):

            # Generate a random initialization file.
            aux.generate_init_file()

            # Simulation
            grmpy.simulate('test.grmpy.ini')

            # Set signal for five seconds.
            signal.alarm(5)

            try:

                # Estimate
                grmpy.estimate(use_simulation=True, init='test.grmpy.ini')

            except TimedOutError:

                pass

            signal.alarm(0)

    @staticmethod
    def test_4():
        """ Testing if the fast and slow evaluation of the criterion function
        result in same value.
        """
        # Initialize containers
        fval = None

        #  Generate a random request with several constraints.
        dict_ = dict()

        dict_['asymptotics'] = 'false'
        dict_['maxiter'] = 0

        dict_ = aux.generate_init_file(dict_)

        # Lock in simulated dataset
        grmpy.simulate('test.grmpy.ini')

        # Loop over fast and slow evaluation of criterion function.
        for version in ['fast', 'slow']:

            # Impose constraints to initialization file
            dict_['ESTIMATION']['version'] = version

            # Print revised initialization file
            aux.print_dict(dict_)

            # Estimate
            rslt = grmpy.estimate(use_simulation=True, init='test.grmpy.ini')

            rslt = rslt.get_attr('max_rslt')

            # Check evaluation result
            if fval is None:
                fval = rslt['fun']
            else:
                assert (np.abs(rslt['fun'] - fval) < SMALL)

    @staticmethod
    def test_5():
        """ Testing if a thousand random initialization requests can be
        processed.
        """

        for _ in range(1000):
            # Generate a random initialization file.
            aux.generate_init_file()


if __name__ == '__main__':
    runmodule()
