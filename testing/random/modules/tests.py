""" This modules contains the tests for the continuous integration efforts.
"""

# standard library
import numpy as np
import signal
import sys
import os

# project library
from modules.exceptions import TimedOutError
import modules.randominit as aux

# Setting up signal handler
def signal_handler(signum, frame):
    raise TimedOutError

signal.signal(signal.SIGALRM, signal_handler)

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
import grmpy

# Module-wide variables
SMALL = 10e-10

''' Main
'''


def test_1():
    """ Testing if a ten random initialization files can be used to simulate a
    model.
    """

    for _ in range(10):

        # Generate a random initialization file.
        aux.generate_init_file()

        # Simulation
        grmpy.simulate('test.grmpy.ini')

        # Finishing
        return True


def test_2():
    """ Testing if a random estimation task can be handled without complaints
    for five seconds.
    """
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

    # Finishing
    return True

def test_3():
    """ Testing if a random estimation task can be handled without complaints
    from beginning till end.
    """
    # Generate a random initialization file.
    aux.generate_init_file()

    # Simulation
    grmpy.simulate('test.grmpy.ini')

    # Estimate
    grmpy.estimate(use_simulation=True, init='test.grmpy.ini')

    # Finishing
    return True


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

    # Finishing
    return True

def test_5():
    """ Testing if a thousand random initialization requests can be simulated
    """

    for _ in range(1000):

        # Generate a random initialization file.
        aux.generate_init_file()

        # Finishing
        return True

