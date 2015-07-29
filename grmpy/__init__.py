""" Interface to public functions of the GRMPY package.
"""

# standard library
import os

# package library
from grmpy.estimate import estimate
from grmpy.simulate import simulate


def test():
    """ Run nose tester for the package.
    """
    base = os.getcwd()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    os.chdir('tests')

    os.system('nosetests test.py')

    os.chdir(base)
