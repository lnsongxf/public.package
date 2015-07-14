""" This module contains functionality related to the estimation.
"""

# standard library
import numdifftools as nd
import numpy as np
import glob
import sys
import os

try:
    import cPickle as pkl
except:
    import pickle as pkl

# project library
from grmpy.tools.auxiliary import _updateParameters
from grmpy.user.init_interface import initialize
from grmpy.clsMax import _scipyWrapperFunction
from grmpy.clsRslt import rsltCls
from grmpy.clsGrm import grmCls
from grmpy.clsMax import maxCls

''' Main function
'''
def estimate(init='init.ini', resume=False, useSimulation=False):
    """ Estimate specified model.
    """
    # Cleanup
    cleanup(resume)

    # Process initialization file
    model_obj, paras_obj, request_obj, _ = initialize(init, useSimulation)

    # Process resume.
    if resume:

        # Antibugging.
        assert (os.path.isfile('info.grmpy.out'))

        # Update parameter objects.
        paras_obj = _updateParameters(paras_obj)

    paras = paras_obj.getValues('internal', 'all')

    # Note starting values
    _write_starting_values(paras)

    # Initialize container
    grm_obj = grmCls()

    grm_obj.setAttr('modelObj', model_obj)

    grm_obj.setAttr('requestObj', request_obj)

    grm_obj.setAttr('parasObj', paras_obj)

    grm_obj.lock()

    # Set random seed.
    np.random.seed(123)

    # Distribute class attributes.
    request_obj = grm_obj.getAttr('requestObj')

    hessian = request_obj.getAttr('hessian')

    with_asymptotics = request_obj.getAttr('withAsymptotics')

    # Distribute auxiliary objects.
    max_obj = maxCls(grm_obj)

    max_obj.lock()

    sys.stdout = open('/dev/null', 'w')

    max_rslt = max_obj.maximize()

    sys.stdout = sys.__stdout__

    # Write optimization results to file
    _write_optimization_results(max_rslt, paras_obj)

    # Distribute results.
    xopt = max_rslt['xopt']

    # Approximate hessian.
    cov_mat = np.tile(np.nan, (len(xopt), len(xopt)))

    if with_asymptotics:
        cov_mat = _add_asymptotics(max_rslt, max_obj, hessian)

    # Construct result class.
    rslt = rsltCls()

    rslt.setAttr('grmObj', grm_obj)

    rslt.setAttr('maxRslt', max_rslt)

    rslt.setAttr('covMat', cov_mat)

    rslt.lock()

    rslt.store('rslt.grmpy.pkl')

    return rslt

''' Auxiliary function
'''
def _add_asymptotics(max_rslt, max_obj, hessian):
    """ Add information about asymptotics.
    """
    # Distribute objects
    xopt = max_rslt['xopt']

    # Construct hessian
    if hessian == 'bfgs':
        cov_mat = max_rslt['covMat']
    elif hessian == 'numdiff':
        crit_func = max_obj.getAttr('critFunc')
        nd_obj = nd.Hessian(lambda x: _scipyWrapperFunction(x, crit_func))
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

def _write_optimization_results(maxRslt, paras_obj):
    """ Write optimization results to file.
    """
    fval = str(maxRslt['fun'])

    if maxRslt['grad'] is not None:
        grad = str(np.amax(np.abs(maxRslt['grad'])))
    else:
        grad = 'None'

    success = str(maxRslt['success'])

    msg = maxRslt['message']

    # Write stop values to file
    paras_obj.update(maxRslt['xopt'], 'external', 'free')

    paras = paras_obj.getValues('internal', 'all')

    with open('info.grmpy.out', 'a') as file_:

        file_.write('''\n  STOP \n\n''')

        for para in paras:

            file_.write('  {:25.18f}'.format(para) + '\n')

    # Write optimization report file
    file_ = open('info.grmpy.out', 'a')

    file_.write('''\n OPTIMIZATION REPORT \n''')

    file_.write('''\n      Function:   ''' + fval)

    file_.write('''\n      Gradient:   ''' + grad + '\n')

    file_.write('''\n      Success:    ''' + success)

    file_.write('''\n      Message:    ''' + msg + '\n\n\n\n')

    file_.close()

def cleanup(resume):
    ''' Cleanup from previous estimation run.
    '''
    # Antibugging.
    assert (resume in [True, False])

    # Construct files list.
    fileList = glob.glob('*.grmpy.*')

    if(resume):

        for file_ in ['info.grmpy.out']:

            try:

                fileList.remove(file_)

            except:

                pass

    # Remove information from simulated data.
    for file_ in ['*.infos.grmpy.out', '*.paras.grmpy.out']:

        try:

            fileList.remove(glob.glob(file_)[0])

        except:

            pass

    # Cleanup
    for file_ in fileList:

        if 'ini' in file_:
            continue

        try:

            os.remove(file_)

        except OSError:

            pass

