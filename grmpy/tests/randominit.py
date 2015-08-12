""" Module that contains the functions for the random generation of an
    initialization file.
"""

# standard library
import numpy as np

''' Module-specific Parameters
'''
MAX_COEFFS = 10
MIN_AGENTS, MAX_AGENTS = 100, 1000

MAX_ITER = 100
MAX_DRAWS = 5

''' Public Function
'''


def generate_init_file(dict_=None):
    """ Get a random initialization file.
    """
    # Antibugging. This interface is using a sentinal value.
    if dict_ is not None:
        assert (isinstance(dict_, dict))

    dict_ = random_dict(dict_)

    print_dict(dict_)

    # Finishing
    return dict_


def random_dict(dict_=None):
    """ Draw random dictionary instance that can be processed into an
        initialization file.
    """
    # Antibugging. This interface is using a sentinal value.
    if dict_ is not None:
        assert (isinstance(dict_, dict))
    else:
        dict_ = dict()

    if 'version' in dict_.keys():
        version = dict_['version']
    else:
        version = np.random.choice(['fast', 'slow'])

    if 'maxiter' in dict_.keys():
        maxiter = dict_['maxiter']
    else:
        maxiter = np.random.random_integers(0, MAX_ITER)

    if 'asymptotics' in dict_.keys():
        asymptotics = dict_['asymptotics']
    else:
        asymptotics = np.random.choice(['true', 'false'])

    if 'optimizer' in dict_.keys():
        optimizer = dict_['optimizer']
    else:
        optimizer = np.random.choice(['bfgs', 'powell'])

    if 'differences' in dict_.keys():
        differences = dict_['differences']
    else:
        differences = np.random.choice(['one-sided', 'two-sided'])

    if 'starts' in dict_.keys():
        starts = dict_['starts']
    else:
        starts = np.random.choice(['manual', 'auto'])

    if 'hess' in dict_.keys():
        hess = dict_['hess']
    else:
        hess = np.random.choice(['bfgs', 'numdiff'])

    if 'AGENTS' in dict_.keys():
        agents = dict_['AGENTS']
    else:
        agents = np.random.random_integers(MIN_AGENTS, MAX_AGENTS)

    ''' Overall
    '''
    num_bene = np.random.random_integers(1, MAX_COEFFS)

    num_cost = np.random.random_integers(1, MAX_COEFFS)

    num_coeffs = 2*num_bene + num_cost + 6 + 2

    positions = list(range(1, num_coeffs + 5))

    num_sim = np.random.random_integers(MIN_AGENTS, MAX_AGENTS)

    constraints = np.random.choice(['!', ' '], size=num_coeffs).tolist()

    # Making sure at least one parameter is up for estimation.s
    if constraints.count(' ') == 0:
        constraints[-1] = ' '

    dict_ = dict()

    ''' DATA
    '''
    dict_['DATA'] = {}
    dict_['DATA']['source'] = 'test.dataset.dat'
    dict_['DATA']['agents'] = agents
    dict_['DATA']['outcome'] = 0
    dict_['DATA']['treatment'] = 1

    ''' BENEFITS
    '''
    dict_['BENE'] = dict()

    # Intercept
    val_bene_1 = round(np.random.ranf(), 2)
    val_bene_2 = round(np.random.ranf(), 2)

    constr_1, constr_2 = constraints.pop(), constraints.pop()
    dict_['BENE']['int'] = [constr_1, val_bene_1, constr_2, val_bene_2]

    # Standard deviation
    val_bene_1 = round(np.random.ranf() + 0.05, 2)
    val_bene_2 = round(np.random.ranf() + 0.05, 2)

    constr_1, constr_2 = constraints.pop(), constraints.pop()
    dict_['BENE']['sd'] = [constr_1, val_bene_1, constr_2, val_bene_2]

    dict_['BENE']['coeff'] = []

    for i in range(num_bene):

        pos, constr_1, constr_2 = positions.pop(), constraints.pop(), \
                                  constraints.pop()
        truth = np.random.choice(['true', 'false'])

        # There needs to be at least one covariate known to the agent at
        # the time of the treatment evaluation.
        if i == 0:
            truth = 'true'

        val_bene_1 = round(np.random.ranf(), 2)
        val_bene_2 = round(np.random.ranf(), 2)

        dict_['BENE']['coeff'] += [[pos, constr_1, val_bene_1, constr_2,
                                  val_bene_2, truth]]

    ''' COSTS
    '''
    dict_['COST'] = dict()

    # Intercept and standard deviation
    constr, val = constraints.pop(), round(np.random.ranf(), 2)
    dict_['COST']['int'] = [constr, val]

    constr, val = constraints.pop(), round(np.random.ranf()*10 + 0.5, 2)
    dict_['COST']['sd'] = [constr, val]

    # Coefficients
    dict_['COST']['coeff'] = []
    for i in range(num_cost):
        pos, constr = positions.pop(), constraints.pop()
        val = round(np.random.ranf(), 2)
        dict_['COST']['coeff'] += [[pos, constr, val]]

    ''' RHO
    '''
    dict_['RHO'] = {}
    dict_['RHO']['untreated'] = [constraints.pop(),
                                 np.random.uniform(-0.98, 0.98)]
    dict_['RHO']['treated'] = [constraints.pop(),
                               np.random.uniform(-0.98, 0.98)]

    # Check that that constraint list is empty
    assert (len(constraints) == 0)

    ''' ESTIMATION
    '''
    dict_['ESTIMATION'] = {}
    dict_['ESTIMATION']['algorithm'] = optimizer
    dict_['ESTIMATION']['maxiter'] = maxiter
    dict_['ESTIMATION']['start'] = starts
    dict_['ESTIMATION']['gtol'] = np.random.uniform(0, 1e-10)
    dict_['ESTIMATION']['epsilon'] = np.random.uniform(0.01, 0.1)
    dict_['ESTIMATION']['differences'] = differences
    dict_['ESTIMATION']['asymptotics'] = asymptotics

    dict_['ESTIMATION']['hessian'] = hess

    if dict_['ESTIMATION']['algorithm'] == 'powell':
        dict_['ESTIMATION']['hessian'] = 'numdiff'

    dict_['ESTIMATION']['alpha'] = np.random.choice([0.01, 0.05, 0.1])
    dict_['ESTIMATION']['draws'] = np.random.random_integers(1, MAX_DRAWS)

    dict_['ESTIMATION']['version'] = version

    ''' SIMULATION
    '''
    dict_['SIMULATION'] = {}

    dict_['SIMULATION']['agents'] = num_sim

    dict_['SIMULATION']['seed'] = np.random.random_integers(1, 100)

    dict_['SIMULATION']['target'] = 'simulation.dat'

    # Make sure some implications are met.
    #
    # As the criterion function is only evaluated at the starting values,
    # there is no call to the BFGS algorithm.
    if (maxiter == 0) and (dict_['ESTIMATION']['asymptotics'] == 'true'):
        dict_['ESTIMATION']['hessian'] = 'numdiff'

    # Finishing.    
    return dict_


def print_dict(dict_):
    """ Generate a random initialization file.
    """
    # Antibugging
    assert (isinstance(dict_, dict))

    # Create initialization
    with open('test.grmpy.ini', 'w') as file_:

        ''' DATA
        '''
        str_ = ' {0:<15} {1:<15} \n'

        file_.write('DATA' + '\n')

        for keys_ in ['source', 'agents']:
            file_.write(str_.format('   ' + keys_, dict_['DATA'][keys_]))

        file_.write('\n')

        for keys_ in ['outcome', 'treatment']:
            file_.write(str_.format('   ' + keys_, dict_['DATA'][keys_]))

        file_.write('\n')

        ''' BENE
        '''
        str_ = ' {0:<8} {1:<5} {2}{3:<3} {4}{5:<3} {6:<6} \n'

        file_.write('BENE' + '\n\n')

        ''' BENE: Coefficients
        '''

        num_coeffs = len(dict_['BENE']['coeff'])

        for i in range(num_coeffs):
            pos, constr_1, val1, constr_2, val2, true = dict_['BENE'][
                'coeff'][i]

            file_.write(
                str_.format('   ' + 'coeff', pos, constr_1, val1, constr_2,
                            val2,
                            true))

        file_.write('\n')

        ''' BENEFITS: Intercept & SD
        '''
        str_ = ' {0:<8} {1:<5} {2}{3:<5} {4}{5:<5} \n'

        for key_ in ['int', 'sd']:
            constr_1, val1, constr_2, val2 = dict_['BENE'][key_]
            file_.write(
                str_.format('   ' + key_, ' ', constr_1, val1, constr_2, val2))

        file_.write('\n')

        ''' COST
        '''

        str_ = ' {0:<8} {1:<5} {2}{3:<5} \n'
        file_.write('COST' + '\n\n')

        num_coeffs = len(dict_['COST']['coeff'])

        for i in range(num_coeffs):
            pos, constr, value = dict_['COST']['coeff'][i]

            file_.write(str_.format('   ' + 'coeff', pos, constr, value))

        file_.write('\n')

        ''' COST: Intercept & SD
        '''
        for key_ in ['int', 'sd']:
            constr, value = dict_['COST'][key_]

            file_.write(str_.format('   ' + key_, ' ', constr, value))

        file_.write('\n')

        ''' RHO
        '''
        str_ = ' {0:<15} {1}{2:<5} \n'

        file_.write('RHO' + '\n\n')

        for key_ in ['treated', 'untreated']:
            constr, value = dict_['RHO'][key_]

            file_.write(str_.format('   ' + key_, constr, value))

        file_.write('\n')

        '''ESTIMATION
        '''
        str_ = ' {0:<15} {1:<15} \n'

        file_.write('ESTIMATION' + '\n\n')

        for key_ in ['algorithm', 'maxiter', 'start', 'gtol']:
            file_.write(str_.format('   ' + key_, dict_['ESTIMATION'][key_]))

        file_.write('\n')

        for key_ in ['epsilon', 'differences']:
            file_.write(str_.format('   ' + key_, dict_['ESTIMATION'][key_]))

        file_.write('\n')

        for key_ in ['asymptotics', 'hessian']:
            file_.write(str_.format('   ' + key_, dict_['ESTIMATION'][key_]))

        file_.write('\n')

        for key_ in ['draws', 'alpha', 'version']:
            file_.write(str_.format('   ' + key_, dict_['ESTIMATION'][key_]))

        file_.write('\n')

        ''' SIMULATION
        '''
        str_ = ' {0:<15} {1:<15} \n'

        file_.write('SIMULATION' + '\n\n')

        for keys_ in ['agents', 'seed', 'target']:
            file_.write(str_.format('   ' + keys_, dict_['SIMULATION'][keys_]))
