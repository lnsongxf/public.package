#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np
import sys
import os

# project library
import modules.battery as development_tests

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
#import grmpy
from grmpy.tests.test import Tests as package_tests


''' Request
'''
label, seed = '0', 484788

''' Error Reproduction
'''
# Define list of admissible tests
test_labels = ['0']
package_labels = ['1', '2', '3', '4', '5']
labels = test_labels + package_labels


if label in test_labels:
    test = getattr(development_tests, 'test_' + label)
elif label in package_labels:
    test = getattr(package_tests, 'test_' + label)
else:
    raise AssertionError

np.random.seed(seed)

# This is required to set the seeds identical to the
# case in the run.py script.
label = np.random.choice(['0', '1', '2', '3', '4', '5'])

test()
