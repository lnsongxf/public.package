""" This module contains a couple of auxiliary functions for the example.
"""

# standard library
import shlex
import numpy as np
import matplotlib.pyplot as plt


def graph_marginal_effects(bmte, cmte, smte):
    """ Plot the marginal effects of treatment in a single graphs.
    """
    # Initialize canvas
    ax = plt.figure(figsize=(12,8)).add_subplot(111, axisbg='white')

    # Plot marginal effects of treatment
    x = np.linspace(0.01, 0.99, 99)
    ax.plot(x, bmte, color='blue', linewidth=4, label=r''' $B^{MTE}$''',
            linestyle='--')
    ax.plot(x, cmte, color='darkgreen', linewidth=4, label=r''' $C^{MTE}$''',
            linestyle='-.')
    ax.plot(x, smte, color='red', linewidth=4, label=r''' $S^{MTE}$''',
            linestyle=':')

    # Set axis labels and ranges
    ax.set_xlabel(r''' $u_S$''', fontsize=20)
    ax.set_ylabel(''' Effect ''', fontsize=20)

    # Set up legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.50, -0.10),
               fancybox=False, frameon=False, shadow=False, ncol=3, fontsize=20)

    # Remove first element on y-axis
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Add title
    plt.suptitle('Marginal Effects of Treatment', fontsize=20)

    # Show plot
    plt.show()

def get_parameters():

    # Initialize containers
    TRUE = []

    # TRUE PARAMETERS
    with open('simulation.infos.grmpy.out', 'r') as file_:

        for line in file_:

            current_line = shlex.split(line)

            if not current_line:
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
