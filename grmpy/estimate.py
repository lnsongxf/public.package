""" This module contains functionality related to the estimation.
"""

# standard library
import glob
import sys
import os

import numdifftools as nd
import numpy as np



# project library
import grmpy.tools.msc as msc
import grmpy.tools.user as user
import grmpy.tools.optimization as opt

from grmpy.clsRslt import RsltCls

''' Main function
'''


def estimate(init='init.ini', resume=False, use_simulation=False):
    """ Estimate specified model.
    """
    # Cleanup
    _cleanup(resume)

    # Process initialization file
    model_obj, paras_obj, _ = user.initialize(init, use_simulation)

    # Update parameter objects.
    if resume:
        paras_obj = msc.update_parameters(paras_obj)

    paras = paras_obj.get_values('internal', 'all')

    # Note starting values
    _write_starting_values(paras)

    # Set random seed
    np.random.seed(123)

    # Distribute class attributes
    hessian = model_obj.get_attr('hessian')

    with_asymptotics = model_obj.get_attr('with_asymptotics')

    # Distribute auxiliary objects
    max_obj = opt.MaxCls(model_obj, paras_obj)

    max_obj.lock()

    stdout_current = sys.stdout

    sys.stdout = open('/dev/null', 'w')

    max_rslt = max_obj.maximize()

    sys.stdout = stdout_current

    # Write optimization results to file
    _write_optimization_results(max_rslt, paras_obj)

    # Distribute results
    xopt = max_rslt['xopt']

    # Approximate hessian
    cov_mat = np.tile(np.nan, (len(xopt), len(xopt)))

    if with_asymptotics:
        cov_mat = _add_asymptotics(max_rslt, max_obj, hessian)

    # Construct result class
    rslt = RsltCls(paras_obj)

    rslt.set_attr('max_rslt', max_rslt)

    rslt.set_attr('cov_mat', cov_mat)

    rslt.lock()

    rslt.store('rslt.grmpy.pkl')

    # Finishing
    return rslt


''' Auxiliary function
'''


def _add_asymptotics(max_rslt, max_obj, hessian):
    """ Add information about asymptotics.
    """
    # Distribute objects
    xopt = max_rslt['xopt']

    # Initialize container
    cov_mat = None

    # Construct hessian
    if hessian == 'bfgs':
        cov_mat = max_rslt['covMat']
    elif hessian == 'numdiff':
        crit_func = max_obj.get_attr('crit_func')
        nd_obj = nd.Hessian(lambda x: opt.scipy_wrapper_function(x, crit_func))
        hess = nd_obj(xopt)
        cov_mat = np.linalg.pinv(hess)

    # Finishing
    return cov_mat


def _write_starting_values(paras):
    """ Write starting values to file.
    """
    with open('info.grmpy.out', 'w') as file_:
        file_.write('''\n  START \n\n''')

        for para in paras:
            file_.write('  {:25.18f}'.format(para) + '\n')


def _write_optimization_results(max_rslt, paras_obj):
    """ Write optimization results to file.
    """
    fval = str(max_rslt['fun'])

    if max_rslt['grad'] is not None:
        grad = str(np.amax(np.abs(max_rslt['grad'])))
    else:
        grad = 'None'

    success = str(max_rslt['success'])

    msg = max_rslt['message']

    # Write stop values to file
    paras_obj.update(max_rslt['xopt'], 'external', 'free')

    paras = paras_obj.get_values('internal', 'all')

    with open('info.grmpy.out', 'a') as file_:

        file_.write('''\n  STOP \n\n''')

        for para in paras:
            file_.write('  {:25.18f}'.format(para) + '\n')

    # Write optimization report file
    with open('info.grmpy.out', 'a') as file_:

        file_.write('''\n OPTIMIZATION REPORT \n''')

        file_.write('''\n      Function:   ''' + fval)

        file_.write('''\n      Gradient:   ''' + grad + '\n')

        file_.write('''\n      Success:    ''' + success)

        file_.write('''\n      Message:    ''' + msg + '\n\n\n\n')


def _cleanup(resume):
    """ Cleanup from previous estimation run.
    """
    # Antibugging.
    assert (resume in [True, False])

    # Construct files list
    file_list = glob.glob('*.grmpy.*')

    if resume:
        file_list.remove('info.grmpy.out')

    # Remove information from simulated data
    for file_ in ['*.infos.grmpy.out']:
        try:
            file_list.remove(glob.glob(file_)[0])
        except IndexError:
            pass

    # Cleanup
    for file_ in file_list:

        if 'ini' in file_:
            continue

        os.remove(file_)
