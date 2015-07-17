#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np
import sys
import os

# project library
import modules.tests as lib
from modules.randominit import *

# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy import *

''' Request
'''
label, seed = 'B', 185423
#label, seed = 'C', 110836

''' Error Reproduction
'''
test = getattr(lib, 'test_' + label)


np.random.seed(seed)

label = np.random.choice(['A', 'B', 'C', 'D', 'E'])

test()
