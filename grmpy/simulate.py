""" Interface to public functions of the grmToolbox.
"""

# standard library
import numpy as np
import random
import os

# project library
from grmpy.tools.msc import *
from grmpy.tools.user import *
from grmpy.tools.optimization import *

''' Main functions
'''
def simulate(init='init.ini', update=False):
    """ Simulate dataset for grmToolbox.
    """

    is_mock = _create_mock(init)

    ''' Process initialization file.
    '''
    _, paras_obj, init_dict = initialize(init, is_simulation=True)

    ''' Distribute information.
    '''
    target = init_dict['SIMULATION']['target']

    seed = init_dict['SIMULATION']['seed']

    np.random.seed(seed)

    random.seed(seed)

    ''' Update parameter class.
    '''
    if update:
        paras_obj = updateParameters(paras_obj)

    ''' Create simulated dataset.
    '''
    if is_mock:
        os.remove(init_dict['DATA']['source'])

    sim_agents = init_dict['SIMULATION']['agents']

    max_ = init_dict['DERIV']['pos']['max']

    sim_dat = np.empty((sim_agents, max_ + 1), dtype='float')

    sim_dat[:, :] = np.nan

    sim_dat = _simulate_exogenous(sim_dat, init_dict)

    sim_dat = _simulate_endogenous(sim_dat, paras_obj, init_dict)

    ''' Update for prediction step.
    '''
    rslt = create_matrices(sim_dat, init_dict)

    paras_obj.unlock()

    paras_obj.set_attr('x_ex_ante', rslt['x_ex_ante'])

    paras_obj.set_attr('x_ex_post', rslt['x_ex_post'])

    paras_obj.lock()

    ''' Save dataset.
    '''
    np.savetxt(target, sim_dat, fmt='%15.10f')

    likl = _get_likelihood(init)

    _write_info(paras_obj, target, rslt, likl)

''' Auxiliary functions.
'''


def _get_likelihood(init):
    """ Calculate likelihood for simulated dataset at true parameter values.
    """
    # Antibugging.
    assert (isinstance(init, str))

    # Process model ingredients.
    model_obj, paras_obj, _ = initialize(init, True)

    # Initialize container.
    crit_obj = CritCls(model_obj, paras_obj)

    crit_obj.lock()

    # Evaluate at true values.
    x = paras_obj.getValues('external', 'free')

    likl = scipy_wrapper_function(x, crit_obj)

    # Cleanup.
    try:
        os.remove('info.grmpy.out')
    except Exception:
        pass

    # Finishing.
    return likl

def _create_mock(init):
    """ Create a mock dataset which allows for use of existing routines
        in the case of a missing source dataset.
    """
    init_dict = processInput(init)

    is_mock = (os.path.exists(init_dict['DATA']['source']) == False)

    obs_agents = init_dict['DATA']['agents']

    pos = init_dict['DATA']['treatment']

    max_ = init_dict['DERIV']['pos']['max']

    sim_dat = np.empty((obs_agents, max_ + 1), dtype='float')

    sim_dat[:, :] = np.random.randn(obs_agents, max_ + 1)

    sim_dat[:, pos] = np.random.random_integers(0, 1, obs_agents)

    source = init_dict['DATA']['source']

    np.savetxt(source, sim_dat, fmt='%15.10f')

    # Finishing.
    return is_mock

def _simulate_endogenous(sim_dat, paras_obj, init_dict):
    """ Simulate the endogenous characteristics such as choices and outcomes.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))
    assert (isinstance(sim_dat, np.ndarray))
    assert (paras_obj.get_status() == True)

    # Distribute information.
    sim_agents = init_dict['SIMULATION']['agents']

    outcome = init_dict['DATA']['outcome']

    treatment = init_dict['DATA']['treatment']

    all_ = init_dict['DERIV']['pos']['all']

    # Sampling of unobservables.
    var_v = paras_obj.getParameters('var',  'V')

    var_u1 = paras_obj.getParameters('var',  'U1')

    var_u0 = paras_obj.getParameters('var',  'U0')

    mean = np.tile(0.0, 3)

    cov_mat = np.diag([var_u1, var_u0, var_v])

    cov_mat[2,0] = cov_mat[0,2] = paras_obj.getParameters('cov', 'U1,V')

    cov_mat[2,1] = cov_mat[1,2] = paras_obj.getParameters('cov', 'U0,V')

    u1, u0, V = np.random.multivariate_normal(mean, cov_mat, sim_agents).T

    # Create data matrices.
    rslt = create_matrices(sim_dat, init_dict)

    x_ex_post = rslt['x_ex_post']

    z = rslt['Z']

    # Simulate choices.
    coeffs_choc = paras_obj.getParameters('choice', None)

    d = (np.dot(coeffs_choc, z.T) - V > 0.0)

    # Potential Outcomes
    outc_treated = paras_obj.getParameters('outc', 'treated')
    outc_untreated = paras_obj.getParameters('outc', 'untreated')

    y1 = np.dot(outc_treated, x_ex_post.T) + u1
    y0 = np.dot(outc_untreated, x_ex_post.T) + u0

    y = d*y1 + (1 - d)*y0

    sim_dat[:, outcome] = y
    sim_dat[:, treatment] = d

    # Quality checks.
    assert (isinstance(sim_dat, np.ndarray))
    assert (np.all(np.isfinite(sim_dat[:, all_])))
    assert (sim_dat.dtype == 'float')

    # Finishing.
    return sim_dat

def _simulate_exogenous(sim_dat, init_dict):
    """ Simulate the exogenous characteristics by filling up the data frame
        with random deviates of the exogenous characteristics from the
        observed dataset.
    """
    # Antibugging.
    assert (isinstance(init_dict, dict))
    assert (isinstance(sim_dat, np.ndarray))

    # Distribute information.
    source = init_dict['DATA']['source']

    sim_agents = init_dict['SIMULATION']['agents']

    all_ = init_dict['DERIV']['pos']['all']

    outcome = init_dict['DATA']['outcome']

    treatment = init_dict['DATA']['treatment']

    # Restrict to exogenous positions.
    for pos in [outcome, treatment]:
        all_.remove(pos)

    # Simulate endogenous characteristics.
    has_source = (os.path.exists(source) == True)

    if has_source:

        obs_dat = np.genfromtxt(source)

        obs_agents = obs_dat.shape[0]

        if obs_agents == sim_agents:

            idx_ = range(obs_agents)

        else:

            idx_ = np.random.randint(0, obs_agents, size = sim_agents)

        for pos in all_:

            sim_dat[:, pos] = obs_dat[idx_, pos]

    else:

        for pos in all_:

            sim_dat[:, pos] = np.random.randn(sim_agents)

    # Quality checks.
    assert (isinstance(sim_dat, np.ndarray))
    assert (np.all(np.isfinite(sim_dat[:, all_])))
    assert (sim_dat.dtype == 'float')

    # Finishing.
    return sim_dat

def _write_info(paras_obj, target, rslt, likl):
    """ Write out some additional infos about the simulated dataset.
    """
    # Auxiliary objects.
    file_name = target.split('.')[0]

    num_agents = str(len(rslt['Y']))

    num_treated = np.sum(rslt['D'] == 1)

    num_untreated = np.sum(rslt['D'] == 0)

    fval = str(likl)

    # Write out structural parameters.
    paras = paras_obj.getValues(version='internal', which='all')

    np.savetxt(file_name + '.paras.grmpy.out', paras, fmt='%15.10f')

    # Write out information on agent experiences.
    with open(file_name + '.infos.grmpy.out', 'w') as file_:

        file_.write('\n Simulated Economy\n\n')

        file_.write('   Number of Observations: ' + num_agents + '\n')

        file_.write('   Function Value:         ' + fval + '\n\n')

        string = '''{0[0]:<10} {0[1]:>12}\n'''

        file_.write('   Choices:  \n\n')

        file_.write(string.format(['     Treated  ', num_treated]))

        file_.write(string.format(['     Untreated', num_untreated]))

        file_.write('\n\n')

        string = '''{0[0]:<10} {0[1]:15.5f}\n'''

        file_.write('   Outcomes:  \n\n')

        file_.write(string.format(['     Treated  ', np.mean(rslt['Y'][rslt['D'] == 1])]))

        file_.write(string.format(['     Untreated', np.mean(rslt['Y'][rslt['D'] == 0])]))

        file_.write('\n\n')