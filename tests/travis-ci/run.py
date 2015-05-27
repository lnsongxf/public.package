""" This module runs the tests.
"""

# standard library
import os


dir_ = os.path.dirname(os.path.realpath(__file__))

os.chdir(dir_)

os.system('python ../testParasCls.py')