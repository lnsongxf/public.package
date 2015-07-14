""" This modules contains the tests for the continuous integration efforts.
"""

# standard library
import numpy as np
import signal
import sys
import os

# project library
from modules.randominit import *
from modules.exceptions import TimedOutError
from modules.auxiliary import perturb

# Setting up signal handler
def signal_handler(signum, frame):
    raise TimedOutError

signal.signal(signal.SIGALRM, signal_handler)

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy import *

# Module-wide variables
SMALL = 10e-10

''' Main
'''


def test_A():
    """ Testing if a ten random initialization files can be used to simulate a model.
    """

    for _ in range(10):

        # Generate a random initialization file.
        generateInitFile()

        # Simulation
        simulate('test.grm.ini')

        # Finishing
        return True


def test_B():
    """ Testing if a random estimation task can be handled without complaints for five seconds.
    """
    # Generate a random initialization file.
    generateInitFile()

    # Simulation
    simulate('test.grm.ini')

    # Perturb
    perturb(scale=0.01, seed=123, init='test.grm.ini', useSimulation=True)

    # Set signal for five seconds.
    signal.alarm(5)

    try:

        # Estimate
        estimate(useSimulation=True, init='test.grm.ini')

    except TimedOutError:

        pass

    signal.alarm(0)

    # Finishing
    return True

def test_C():
    """ Testing if the perturbation function works properly
    """
    # Generate a random initialization file.
    generateInitFile()

    # Simulation
    simulate('test.grm.ini')

    # Perturb
    for _ in range(100):

        # Draw scale
        scale = np.random.random()

        # Perturb parameter values
        perturb(scale=scale, seed=123, init='test.grm.ini', useSimulation=True)

    # Finishing
    return True

def test_D():
    """ Testing if a random estimation task can be handled without complaints from beginning till end.
    """
    # Generate a random initialization file.
    generateInitFile()

    # Simulation
    simulate('test.grm.ini')

    # Perturb
    perturb(scale=0.01, seed=123, init='test.grm.ini', useSimulation=True)

    # Estimate
    estimate(useSimulation=True, init='test.grm.ini')

    # Finishing
    return True


def test_E():
    """ Testing if the fast and slow evaluation of the criterion function result in same value.
    """
    # Get a seeed
    seed = np.random.randint(1)

    # Initialize containers
    fval = None

    # Loop over fast and slow evaluation of criterion function.
    for version in ['fast', 'slow']:

        np.random.seed(seed)

        # Impose constraints to initialization file
        dict_ = dict()
        dict_['maxiter'] = 0
        dict_['version'] = version
        dict_['asymptotics'] = 'false'

        # Generate random request
        generateInitFile(dict_)

        # Simulation
        simulate('test.grm.ini')

        # Estimate
        rslt = estimate(useSimulation=True, init='test.grm.ini')

        rslt = rslt.getAttr('maxRslt')

        # Check evaluation result
        if fval is None:
            fval = rslt['fun']
        else:
            assert (np.abs(rslt['fun'] - fval) < SMALL)

    # Finishing
    return True

def test_F():
    """ Testing if a thousand random initialization requests can be simulated
    """

    for _ in range(1000):

        # Generate a random initialization file.
        generateInitFile()

        # Finishing
        return True

