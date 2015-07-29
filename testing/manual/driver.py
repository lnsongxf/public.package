#!/usr/bin/env python
""" This module serves as the interface for manual tests of the GRMPY package.
"""

# standard library
import numpy as np
np.set_printoptions(precision=4)

import shlex
import sys
import os


# GRMPY import
sys.path.insert(0, os.environ['GRMPY'])
from grmpy import *

""" Auxiliary functions
"""

def get_parameters():

    # Initialize containers
    TRUE = []

    # TRUE PARAMETERS
    with open('simulation.infos.grmpy.out', 'r') as file_:

        for line in file_:

            current_line = shlex.split(line)

            if len(current_line) == 0:
                continue

            if current_line[0].isupper():
                keyword = current_line[0]
                continue

            if keyword == 'TRUE':
                TRUE += [np.float(current_line[0])]

    # START and STOP parameters
    START, STOP = [], []

    with open('info.grmpy.out', 'r') as file_:

        for line in file_:

            current_line = shlex.split(line)

            if not current_line:
                continue

            if current_line[0].isupper():
                keyword = current_line[0]
                continue

            if keyword == 'START':
                START += [np.float(current_line[0])]

            if keyword == 'STOP':
                STOP += [np.float(current_line[0])]

    # Type transformations
    START = np.array(START)
    STOP = np.array(STOP)
    TRUE = np.array(TRUE)

    # Finishing
    return START, STOP, TRUE



simulate('test.grmpy.ini')

estimate(use_simulation=True, init='test.grmpy.ini')


str_ = '{0:10.2f}{1:10.2f}{2:10.2f}{3:10.2f}'

START, STOP, TRUE = get_parameters()

for i in range(len(STOP)):

    print str_.format(START[i], STOP[i], TRUE[i], np.abs(STOP[i] - TRUE[i]))
