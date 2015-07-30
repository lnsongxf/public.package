""" This modules contains some additional tests that are only used in
long-run development tests.
"""

# standard library
import numpy as np
import signal
import sys
import os

# project library
import modules.randominit as aux

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
import grmpy

''' Main
'''
def test_0():
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
