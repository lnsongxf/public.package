""" Module for auxiliary functions that are used throughout the GRMPY package.
"""

# standard library
import numpy as np
import shlex
import os


def update_parameters(paras_obj):
    """ Update parameter object if possible.
    """
    # Antibugging 
    assert (paras_obj.get_status() is True)
    assert (os.path.isfile('info.grmpy.out'))

    # Update
    has_step = (os.path.isfile('info.grmpy.out'))

    if has_step:

        list_ = []

        is_relevant = False

        with open('info.grmpy.out', 'r') as file_:

            for line in file_:

                current_line = shlex.split(line)

                if len(current_line) == 0:
                    continue

                if len(current_line) > 1:
                    break

                if current_line == ['STOP']:
                    is_relevant = True

                if current_line[0] in ['START', 'STOP']:
                    continue

                if is_relevant:
                    list_ += [np.float(current_line[0])]

        starting_values = np.array(list_)

        paras_obj.update(starting_values, 'internal', 'all')

    # Finishing
    return paras_obj


def create_matrices(dataset, init_dict):
    """ Create the data matrices.
    """
    # Antibugging 
    assert (isinstance(init_dict, dict))
    assert (isinstance(dataset, np.ndarray))
    assert (dataset.dtype == 'float')
    assert (dataset.ndim == 2)
    
    # Distribute information
    outcome = init_dict['DATA']['outcome']
    treatment = init_dict['DATA']['treatment']
    common = init_dict['DERIV']['common']['pos']
    excl_bene_ex_ante = init_dict['DERIV']['excl_bene']['ex_ante']['pos']
    excl_bene_ex_post = init_dict['DERIV']['excl_bene']['ex_post']['pos']
    excl_cost = init_dict['DERIV']['excl_cost']['pos']

    # Construct auxiliary information 
    num_agents = dataset.shape[0]
    
    # Create matrices 
    y = dataset[:, outcome]
    d = dataset[:, treatment]
    m = dataset[:, common].copy()
    m = np.concatenate((m, np.ones((num_agents, 1))), axis=1)
    x_ex_ante = np.concatenate((dataset[:, excl_bene_ex_ante], m), axis=1)
    x_ex_post = np.concatenate((dataset[:, excl_bene_ex_post], m), axis=1)
    g = np.concatenate((m, dataset[:, excl_cost]), axis=1)
    z = np.concatenate((x_ex_ante, dataset[:, excl_cost]), axis=1)
    
    # Quality checks 
    for mat in [x_ex_ante, x_ex_post, g, z]:
        assert (isinstance(mat, np.ndarray))
        assert (mat.dtype == 'float')
        assert (mat.ndim == 2)
    
    for mat in [d, y]:
        assert (isinstance(mat, np.ndarray))
        assert (mat.dtype == 'float')
        assert (mat.ndim == 1)

    # Collect 
    rslt = dict()
    rslt['x_ex_post'] = x_ex_post
    rslt['x_ex_ante'] = x_ex_ante
    rslt['G'] = g
    rslt['Z'] = z
    rslt['Y'] = y
    rslt['D'] = d
        
    # Finishing.
    return rslt






