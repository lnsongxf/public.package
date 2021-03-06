""" Interface to public functions of the grmToolbox.
"""

# standard library
import random
import os

import numpy as np



# project library
import grmpy.tools.msc as msc
import grmpy.tools.user as user
import grmpy.tools.optimization as opt

''' Main functions
'''


def simulate(init='init.ini', update=False):
    """ Simulate dataset for grmToolbox.
    """

    is_mock = _create_mock(init)

    # Process initialization file
    _, paras_obj, init_dict = user.initialize(init, is_simulation=True)

    # Distribute information
    target = init_dict['SIMULATION']['target']

    seed = init_dict['SIMULATION']['seed']

    np.random.seed(seed)

    random.seed(seed)

    # Update parameter class
    if update:
        paras_obj = msc.update_parameters(paras_obj)

    # Create simulated dataset
    if is_mock:
        os.remove(init_dict['DATA']['source'])

    sim_agents = init_dict['SIMULATION']['agents']

    max_ = init_dict['DERIV']['pos']['max']

    sim_dat = np.empty((sim_agents, max_ + 1), dtype='float')

    sim_dat[:, :] = np.nan

    sim_dat = _simulate_exogenous(sim_dat, init_dict)

    sim_dat = _simulate_endogenous(sim_dat, paras_obj, init_dict)

    # Update for prediction step
    rslt = msc.create_matrices(sim_dat, init_dict)

    paras_obj.unlock()

    paras_obj.set_attr('X_ex_post', rslt['X_ex_post'])

    paras_obj.set_attr('X_ex_ante', rslt['X_ex_ante'])

    paras_obj.lock()

    # Save dataset
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

    # Process model ingredients
    model_obj, paras_obj, _ = user.initialize(init, True)

    # Initialize container
    crit_obj = opt.CritCls(model_obj, paras_obj)

    crit_obj.lock()

    # Evaluate at true values
    x = paras_obj.get_values('external', 'free')

    likl = opt.scipy_wrapper_function(x, crit_obj)

    # Cleanup
    try:
        os.remove('info.grmpy.out')
    except OSError:
        pass

    # Finishing.
    return likl


def _create_mock(init):
    """ Create a mock dataset which allows for use of existing routines
        in the case of a missing source dataset.
    """
    init_dict = user.process_input(init)

    is_mock = (os.path.exists(init_dict['DATA']['source']) == False)

    obs_agents = init_dict['DATA']['agents']

    pos = init_dict['DATA']['treatment']

    max_ = init_dict['DERIV']['pos']['max']

    sim_dat = np.empty((obs_agents, max_ + 1), dtype='float')

    sim_dat[:, :] = np.random.randn(obs_agents, max_ + 1)

    sim_dat[:, pos] = np.random.random_integers(0, 1, obs_agents)

    source = init_dict['DATA']['source']

    np.savetxt(source, sim_dat, fmt='%15.10f')

    # Finishing
    return is_mock


def _simulate_endogenous(sim_dat, paras_obj, init_dict):
    """ Simulate the endogenous characteristics such as choices and outcomes.
    """
    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (isinstance(sim_dat, np.ndarray))
    assert (paras_obj.get_status() is True)

    # Distribute information
    sim_agents = init_dict['SIMULATION']['agents']

    outcome = init_dict['DATA']['outcome']

    treatment = init_dict['DATA']['treatment']

    all_ = init_dict['DERIV']['pos']['all']

    # Sampling of unobservables
    var_v = paras_obj.get_parameters('var', 'V')

    var_u1 = paras_obj.get_parameters('var', 'U1')

    var_u0 = paras_obj.get_parameters('var', 'U0')

    mean = np.tile(0.0, 3)

    cov_mat = np.diag([var_u1, var_u0, var_v])

    cov_mat[2, 0] = cov_mat[0, 2] = paras_obj.get_parameters('cov', 'U1,V')

    cov_mat[2, 1] = cov_mat[1, 2] = paras_obj.get_parameters('cov', 'U0,V')

    u1, u0, v = np.random.multivariate_normal(mean, cov_mat, sim_agents).T

    # Create data matrices
    rslt = msc.create_matrices(sim_dat, init_dict)

    x = rslt['X_ex_post']

    z = rslt['Z']

    # Simulate choices
    coeffs_choc = paras_obj.get_parameters('choice', None)

    d = (np.dot(coeffs_choc, z.T) - v > 0.0)

    # Potential outcomes
    outc_treated = paras_obj.get_parameters('outc', 'treated')
    outc_untreated = paras_obj.get_parameters('outc', 'untreated')

    y1 = np.dot(outc_treated, x.T) + u1
    y0 = np.dot(outc_untreated, x.T) + u0

    y = d * y1 + (1 - d) * y0

    sim_dat[:, outcome] = y
    sim_dat[:, treatment] = d

    # Quality checks
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
    # Antibugging
    assert (isinstance(init_dict, dict))
    assert (isinstance(sim_dat, np.ndarray))

    # Distribute information
    source = init_dict['DATA']['source']

    sim_agents = init_dict['SIMULATION']['agents']

    all_ = init_dict['DERIV']['pos']['all']

    outcome = init_dict['DATA']['outcome']

    treatment = init_dict['DATA']['treatment']

    # Restrict to exogenous positions
    for pos in [outcome, treatment]:
        all_.remove(pos)

    # Simulate endogenous characteristics
    has_source = (os.path.exists(source) is True)

    if has_source:

        obs_dat = np.genfromtxt(source)

        obs_agents = obs_dat.shape[0]

        if obs_agents == sim_agents:

            idx_ = range(obs_agents)

        else:

            idx_ = np.random.randint(0, obs_agents, size=sim_agents)

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
    """ Write out some additional info about the simulated dataset.
    """
    # Auxiliary objects
    file_name = target.split('.')[0]

    num_agents = str(len(rslt['Y']))

    num_treated = np.sum(rslt['D'] == 1)

    num_untreated = np.sum(rslt['D'] == 0)

    fval = str(likl)

    paras = paras_obj.get_values('internal', 'all')

    # Write out information on agent experiences
    with open(file_name + '.infos.grmpy.out', 'w') as file_:
        file_.write('\n SIMULATED DATASET\n\n')

        file_.write('   Number of Observations  ' + num_agents + '\n')

        file_.write('   Function Value          ' + fval + '\n\n')

        string = '''{0[0]:<10} {0[1]:>11}\n'''

        file_.write('   Treatment Status:  \n\n')

        file_.write(string.format(['     Treated  ', num_treated]))

        file_.write(string.format(['     Untreated', num_untreated]))

        file_.write('\n\n')

        string = '''{0[0]:<10} {0[1]:15.5f}\n'''

        file_.write('   Average Outcomes by Treatment Status:  \n\n')

        file_.write(string.format(
            ['     Treated  ', np.mean(rslt['Y'][rslt['D'] == 1])]))

        file_.write(string.format(
            ['     Untreated', np.mean(rslt['Y'][rslt['D'] == 0])]))

        file_.write('\n\n')

        # Parameters
        file_.write('''\n  TRUE PARAMETERS \n\n''')

        for para in paras:
            file_.write('  {:25.18f}'.format(para) + '\n')
