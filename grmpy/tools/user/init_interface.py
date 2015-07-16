""" Interface for all routines related to the processing of the
    initialization file.
"""

# standard library
import numpy as np
import shlex
import os

# project library
from grmpy.tools.user.create_model import *
from grmpy.tools.user.create_paras import *
from grmpy.tools.user.check_input import *

''' Main function.
'''
def initialize(init_file, use_simulation = False, is_simulation = False):
    """ Read initialization file and construct the objects required for the
        estimation runs.
    """
    # Antibugging
    assert (os.path.exists(init_file))
    
    # Process initialization file
    init_dict = process_input(init_file)
    
    # Use SIMULATION info
    if use_simulation:
        init_dict['DATA']['source'] = init_dict['SIMULATION']['target']
        init_dict['DATA']['agents'] = init_dict['SIMULATION']['agents']
    
    # Construct objects
    model_obj = construct_model(init_dict)
    paras_obj = construct_paras(init_dict, model_obj, is_simulation)

    # Quality checks
    for obj in [model_obj, paras_obj]:
        assert (obj.get_status() is True)
    
    # Finishing
    return model_obj, paras_obj, init_dict

def process_input(init_file):
    """ Create dictionary from information in initialization file.
    """
    # Antibugging.
    assert (os.path.exists(init_file))

    init_dict = _construct_dictionary()

    with open(init_file, 'r') as init_file:

        for line in init_file:

            current_line = shlex.split(line)

            ''' Preprocessing.
            '''
            is_empty, is_keyword = _process_cases(current_line)

            if is_empty:
                continue

            elif is_keyword:
                keyword = current_line[0]
                continue

            ''' Process major blocks.
            '''
            if keyword ==  'DATA':

                init_dict = _process_data(init_dict, current_line)

            if keyword == 'BENE':

                init_dict = _process_bene(init_dict, current_line)

            if keyword == 'COST':

                init_dict = _process_cost(init_dict, current_line)

            if keyword ==  'RHO':

                init_dict = _process_rho(init_dict, current_line)

            if keyword == 'ESTIMATION':

                init_dict = _process_estimation(init_dict, current_line)

            if keyword == 'SIMULATION':

                init_dict = _process_simulation(init_dict, current_line)

    # Add derived information.
    init_dict = _add_deriv(init_dict)

    # Check quality.
    assert (check_input(init_dict) == True)

    # Type transformation.
    init_dict = _type_transformations(init_dict)

    # Finishing.
    return init_dict

''' Private auxiliary functions.
'''
def _add_deriv(init_dict):
    """ Add useful derived information for easy access.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))

    # Additional information
    init_dict['DERIV'] = {}

    ''' All and maximum position of covariates.
    '''
    # Initialization
    init_dict['DERIV']['pos'] = {}

    init_dict['DERIV']['pos']['max'] = None

    init_dict['DERIV']['pos']['all'] = None

    # Distribute source information
    y_pos = init_dict['DATA']['outcome']

    d_pos = init_dict['DATA']['treatment']

    b_pos = init_dict['BENE']['TREATED']['coeffs']['pos']

    g_pos = init_dict['COST']['coeffs']['pos']

    # Construct derived information
    all_ = list(set([y_pos] + [d_pos] + b_pos + g_pos))

    max_ = int(max(all_))

    # Collect
    init_dict['DERIV']['pos']['max'] = max_

    init_dict['DERIV']['pos']['all'] = all_

    ''' Position and number of exclusions and common elements.
    '''
    # Initialization
    init_dict['DERIV']['common'] = {}

    init_dict['DERIV']['common']['pos'] = []

    init_dict['DERIV']['common']['num'] = None

    init_dict['DERIV']['excl_bene'] = {}

    init_dict['DERIV']['excl_bene']['ex_ante'] = {}

    init_dict['DERIV']['excl_bene']['ex_post'] = {}

    init_dict['DERIV']['excl_bene']['ex_post']['pos'] = []

    init_dict['DERIV']['excl_bene']['ex_post']['num'] = None

    init_dict['DERIV']['excl_bene']['ex_ante']['pos'] = []

    init_dict['DERIV']['excl_bene']['ex_ante']['num'] = None

    init_dict['DERIV']['excl_cost'] = {}

    init_dict['DERIV']['excl_cost']['pos'] = {}

    init_dict['DERIV']['excl_cost']['num'] = None

    # Distribute source information
    bene = dict()

    bene['pos'] = np.array(init_dict['BENE']['TREATED']['coeffs']['pos'])

    bene['info'] = np.array(init_dict['BENE']['TREATED']['coeffs']['info'])

    cost = dict()

    cost['pos'] = np.array(init_dict['COST']['coeffs']['pos'])

    # Construct auxiliary objects
    no_covariates = (len(bene['info']) == 0)

    # Construct derived information
    init_dict['DERIV']['common']['pos'] = \
        list(set(bene['pos']).intersection(cost['pos']))

    if no_covariates:
        init_dict['DERIV']['excl_bene']['ex_ante']['pos'] = []

    else:

        init_dict['DERIV']['excl_bene']['ex_ante']['pos'] = \
            list(set(bene['pos'][bene['info']]).difference(cost['pos']))

    init_dict['DERIV']['excl_bene']['ex_post']['pos'] = \
        list(set(bene['pos']).difference(cost['pos']))

    init_dict['DERIV']['excl_cost']['pos'] = \
        list(set(cost['pos']).difference(bene['pos']))

    init_dict['DERIV']['common']['num'] = \
        len(init_dict['DERIV']['common']['pos'])

    init_dict['DERIV']['excl_bene']['ex_ante']['num'] = \
        len(init_dict['DERIV']['excl_bene']['ex_ante']['pos'])

    init_dict['DERIV']['excl_bene']['ex_post']['num'] = \
        len(init_dict['DERIV']['excl_bene']['ex_post']['pos'])

    init_dict['DERIV']['excl_cost']['num'] = \
        len(init_dict['DERIV']['excl_cost']['pos'])

    # Finishing.
    return init_dict

def _type_transformations(init_dict):
    """ Type transformations
    """
    # Antibugging
    assert (isinstance(init_dict, dict))

    # Type conversions.
    for subgroup in ['TREATED', 'UNTREATED']:
        init_dict['BENE'][subgroup]['coeffs']['info'] = \
            np.array(init_dict['BENE'][subgroup]['coeffs']['info'])

    # Finishing
    return init_dict

def _construct_dictionary():
    """ Construct dictionary from initialization file.
    """

    # Initialize dictionary keys
    init_dict = dict()

    init_dict['BENE'] = {}
    init_dict['COST'] = {}

    init_dict['COST']['coeffs'] = {}
    init_dict['COST']['int'] = {}
    init_dict['COST']['sd'] = {}

    init_dict['COST']['coeffs']['values'] = []
    init_dict['COST']['coeffs']['pos'] = []
    init_dict['COST']['coeffs']['free'] = []


    init_dict['COST']['sd']['values'] = []
    init_dict['COST']['sd']['free'] = []

    init_dict['COST']['int']['values'] = []
    init_dict['COST']['int']['free'] = []

    init_dict['BENE']['TREATED'] = {}
    init_dict['BENE']['UNTREATED'] = {}

    init_dict['BENE']['TREATED'] = {}

    init_dict['BENE']['TREATED']['coeffs'] = {}
    init_dict['BENE']['TREATED']['sd'] = {}

    init_dict['BENE']['TREATED']['coeffs']['values'] = []
    init_dict['BENE']['TREATED']['coeffs']['pos'] = []
    init_dict['BENE']['TREATED']['coeffs']['info'] = []
    init_dict['BENE']['TREATED']['coeffs']['free'] = []

    init_dict['BENE']['TREATED']['int'] = {}
    init_dict['BENE']['TREATED']['int']['values'] = []
    init_dict['BENE']['TREATED']['int']['free'] = []

    init_dict['BENE']['TREATED']['sd'] = {}
    init_dict['BENE']['TREATED']['sd']['values'] = []
    init_dict['BENE']['TREATED']['sd']['free'] = []

    init_dict['BENE']['UNTREATED']['coeffs'] = {}
    init_dict['BENE']['UNTREATED']['sd'] = {}

    init_dict['BENE']['UNTREATED']['coeffs']['values'] = []
    init_dict['BENE']['UNTREATED']['coeffs']['pos'] = []
    init_dict['BENE']['UNTREATED']['coeffs']['info'] = []
    init_dict['BENE']['UNTREATED']['coeffs']['free'] = []

    init_dict['BENE']['UNTREATED']['int'] = {}
    init_dict['BENE']['UNTREATED']['int']['values'] = []
    init_dict['BENE']['UNTREATED']['int']['free'] = []

    init_dict['BENE']['UNTREATED']['sd'] = {}
    init_dict['BENE']['UNTREATED']['sd']['values'] = []
    init_dict['BENE']['UNTREATED']['sd']['free'] = []

    init_dict['DATA'] = {}

    init_dict['RHO'] = {}
    init_dict['RHO']['treated'] = {}
    init_dict['RHO']['untreated'] = {}

    init_dict['ESTIMATION'] = {}

    init_dict['SIMULATION'] = {}

    return init_dict

def _process_cases(current_line):
    """ Process special cases of empty list and keywords.
    """
    def _check_empty(current_line):
        """ Check whether the list is empty.
        """
        # Antibugging
        assert (isinstance(current_line, list))

        # Evaluate list
        is_empty = (len(current_line) == 0)

        # Check integrity
        assert (isinstance(is_empty, bool))

        # Finishing 
        return is_empty

    def _check_keyword(current_line):
        """ Check for keyword.
        """
        # Antibugging.
        assert (isinstance(current_line, list))

        # Evaluate list.
        is_keyword = False

        if len(current_line) > 0:

            is_keyword = (current_line[0].isupper())

        # Check integrity
        assert (isinstance(is_keyword, bool))

        # Finishing
        return is_keyword

    ''' Main Function.
    '''
    # Antibugging
    assert (isinstance(current_line, list))

    # Determine indicators.
    is_empty = _check_empty(current_line)

    is_keyword = _check_keyword(current_line)

    # Finishing
    return is_empty, is_keyword

''' Processing of major blocks.
'''
def _process_bene(init_dict, current_line):
    """ Process BENE block.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (isinstance(current_line, list))

    # Process information
    type_ = current_line[0]

    assert (type_ in ['coeff', 'int', 'sd'])

    if type_ == 'coeff':

        pos = current_line[1]

        assert (len(current_line) == 5)

        assert (current_line[4].upper() in ['TRUE', 'FALSE'])

        info = (current_line[4].upper() == 'TRUE')

        is_free = (current_line[2][0] != '!')
        value = current_line[2].replace('!','')

        init_dict['BENE']['TREATED']['coeffs']['values'] += [float(value)]
        init_dict['BENE']['TREATED']['coeffs']['free'] += [is_free]

        is_free = (current_line[3][0] != '!')
        value = current_line[3].replace('!','')

        init_dict['BENE']['UNTREATED']['coeffs']['values'] += [float(value)]
        init_dict['BENE']['UNTREATED']['coeffs']['free'] += [is_free]

        for subgroup in ['TREATED', 'UNTREATED']:
            init_dict['BENE'][subgroup]['coeffs']['info'] += [info]
            init_dict['BENE'][subgroup]['coeffs']['pos'] += [int(pos)]

    if type_ in ['sd', 'int']:

        assert (len(current_line) == 3)

        is_free = (current_line[1][0] != '!')
        value = current_line[1].replace('!','')

        init_dict['BENE']['TREATED'][type_]['values'] += [float(value)]
        init_dict['BENE']['TREATED'][type_]['free'] += [is_free]

        is_free = (current_line[2][0] != '!')
        value = current_line[2].replace('!','')

        init_dict['BENE']['UNTREATED'][type_]['values'] += [float(value)]
        init_dict['BENE']['UNTREATED'][type_]['free'] += [is_free]

    # Finishing.
    return init_dict

def _process_cost(init_dict, current_line):
    """ Process COST block.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (isinstance(current_line, list))

    # Process information
    type_ = current_line[0]

    assert (type_ in ['coeff', 'int', 'sd'])

    if type_ == 'coeff':

        assert (len(current_line) == 3)

        pos = current_line[1]
        is_free = (current_line[2][0] != '!')
        value = current_line[2].replace('!','')

        init_dict['COST']['coeffs']['values'] += [float(value)]
        init_dict['COST']['coeffs']['pos'] += [int(pos)]
        init_dict['COST']['coeffs']['free'] += [is_free]

    if type_ in ['sd', 'int']:

        assert (len(current_line) == 2)

        is_free = (current_line[1][0] != '!')
        value = current_line[1].replace('!','')

        init_dict['COST'][type_]['values'] += [float(value)]
        init_dict['COST'][type_]['free'] += [is_free]

    # Finishing.
    return init_dict

def _process_rho(init_dict, current_line):
    """ Process RHO block.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))
    assert (isinstance(current_line, list))
    assert (len(current_line) == 2)

    # Process information.
    assert (current_line[0] in ['untreated', 'treated'])

    name = current_line[0]

    is_free = (current_line[1][0] != '!')
    value = current_line[1].replace('!','')

    if name not in init_dict['RHO'].keys():

        init_dict['RHO'][name] = {}

    init_dict['RHO'][name]['value'] = float(value)
    init_dict['RHO'][name]['free'] = is_free

    # Finishing.
    return init_dict

def _process_estimation(init_dict, current_line):
    """ Process ESTIMATION block.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))
    assert (isinstance(current_line, list))
    assert (len(current_line) == 2)

    # Process information.
    keyword = current_line[0]
    flag = current_line[1]

    # Special treatments.
    if keyword in ['gtol', 'epsilon']:
        flag = float(flag)

    if keyword == 'maxiter':
        if flag.upper() == 'NONE':
            flag = None
        else:
            flag = int(flag)

    if keyword in ['asymptotics']:
        assert (flag.upper() in ['TRUE', 'FALSE'])
        if flag.upper() == 'TRUE':
            flag = True
        else:
            flag = False

    # Special treatments.
    if keyword == 'alpha':
        flag = float(flag)

    if keyword in ['draws']:
        flag = int(flag)

    # Construct dictionary.
    init_dict['ESTIMATION'][keyword] = flag

    # Finishing.
    return init_dict

def _process_simulation(init_dict, current_line):
    """ Process SIMULATION block.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (isinstance(current_line, list))
    assert (len(current_line) == 2)

    # Process information
    keyword = current_line[0]
    flag = current_line[1]

    # Special treatments
    if keyword in ['agents', 'seed']:
        flag = int(flag)

    # Construct dictionary.
    init_dict['SIMULATION'][keyword] = flag

    # Finishing.
    return init_dict

def _process_data(init_dict, current_line):
    """ Process DATA block.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (isinstance(current_line, list))
    assert (len(current_line) == 2)

    # Process information
    keyword = current_line[0]
    flag = current_line[1]

    # Special treatments
    if keyword in ['outcome', 'treatment']:
        flag = int(flag)

    if keyword == 'agents':
        if flag.upper() == 'NONE':
            flag = None
        else:
            flag = int(flag)

    # Construct dictionary
    init_dict['DATA'][keyword] = flag

    # Finishing.
    return init_dict

