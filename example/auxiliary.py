""" This module contains a couple of auxiliary functions for the example.
"""

# standard library
import shlex
import numpy as np

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

            if len(current_line) == 0:
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
