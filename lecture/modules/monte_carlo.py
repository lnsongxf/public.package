#!/usr/bin/env python
""" This module conducts the Monte Carlo exercise discussed in the
    lecture.
"""

# standard library
import os
import argparse
import numpy as np

''' Main functions
'''
def conduct_monte_carlo(init_file):
    """ This function conducts a Monte Carlo exercise to test the
        reliabilty of the grmToolbox.
    """

    # Clean directory
    os.system('grmToolbox-clean')

    # Estimate generalized Roy model on DATA/source
    os.system('grmToolbox-estimate')

    # Simulate dataset with perturbed parameter values
    # and store it as SIMULATION/target
    os.system('grmToolbox-simulate --init init.ini --update')

    # Reestimate generalize Roy model using SIMULATION
    # as source and starting values from initialization
    # file
    os.system('grmToolbox-estimate --init init.ini --simulation')


def process(args):
    """ Process arguments.
    """
    # Distribute arguments
    init_file = args.init_file

    # Quality checks
    assert (isinstance(init_file, str))

    # Finishing
    return init_file


def print_results():
    """ Print results from Monte Carlo Exercise.
    """

    # Load true and estimated parameters
    true_values = np.loadtxt('simulation.paras.grm.out')
    est_values = np.loadtxt('stepParas.grm.out')
    start_values = np.loadtxt('startParas.grm.out')

    # Auxiliary objects
    num_paras = len(true_values)

    # Formatting
    fmt = '{0:10.2f}{1:10.2f}{2:10.2f}{3:10.2f}'

    # Print both parameters
    print '     Start  Estimate     Truth    Difference \n'
    for i in range(num_paras):
        start, est, true = start_values[i], est_values[i], true_values[i]

        diff = est - true

        print fmt.format(start, est, true, diff)


''' Execution of module as script.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        'Conduct Monte Carlo exercise with grmToolbox.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init', action='store', dest='init_file',
        default='init.ini', help='source for model configuration')

    args = parser.parse_args()

    init_file = process(args)

    conduct_monte_carlo(init_file)

    print_results()
