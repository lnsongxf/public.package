#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np
import sys
import os

# project library
import modules.tests as lib

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy import *

''' Request
'''
label, seed = '3', 110836

''' Error Reproduction
'''
test = getattr(lib, 'test_' + label)

np.random.seed(seed)

# This is required to set the seeds identical to the
# case in the run.py script.
label = np.random.choice(['1', '2', '3', '4', '5'])

test()
