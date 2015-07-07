#!/usr/bin/env python
""" Script to quickly investigate failed estimation runs.
"""
# standard library
import numpy as np

# project library
import modules.tests as lib

''' Request
'''
label, seed = 'E', 96222

''' Error Reproduction
'''

test = getattr(lib, 'test_' + label)

np.random.seed(seed)

test()