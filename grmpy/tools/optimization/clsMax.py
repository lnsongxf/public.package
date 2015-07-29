""" Module for the management of the maximization algorithms.
"""

# standard library.
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_powell

import numpy as np

# project library
from grmpy.tools.optimization.wrappers import scipy_wrapper_function
from grmpy.tools.optimization.wrappers import scipy_wrapper_gradient
from grmpy.tools.optimization.clsCrit import CritCls

from grmpy.clsMeta import MetaCls
from grmpy.clsModel import ModelCls
from grmpy.clsParas import ParasCls


class MaxCls(MetaCls):
    def __init__(self, model_obj, paras_obj):

        # Antibugging
        assert (isinstance(model_obj, ModelCls))
        assert (isinstance(paras_obj, ParasCls))

        assert (model_obj.get_status() is True)
        assert (paras_obj.get_status() is True)

        # Attributes
        self.attr = dict()
        self.attr['model_obj'] = model_obj
        self.attr['paras_obj'] = paras_obj

        # Results container
        self.attr['max_rslt'] = None

        # Status
        self.is_locked = False

    def maximize(self):
        """ Maximization
        """
        # Antibugging
        assert (self.get_status() is True)

        # Distribute class attributes
        crit_func = self.get_attr('crit_func')
        paras_obj = self.get_attr('paras_obj')
        model_obj = self.get_attr('model_obj')
        algorithm = model_obj.get_attr('algorithm')
        maxiter = model_obj.get_attr('maxiter')

        # Maximization
        max_rslt = None

        if maxiter == 0:
            x = paras_obj.get_values('external', 'free')

            max_rslt = dict()
            max_rslt['fun'] = scipy_wrapper_function(x, crit_func)
            max_rslt['grad'] = scipy_wrapper_gradient(x, crit_func)
            max_rslt['xopt'] = x
            max_rslt['success'] = False

            # Message:
            max_rslt['message'] = 'Single function evaluation at ' \
                                  'starting values.'

        elif algorithm == 'bfgs':
            max_rslt = self._bfgs()

        elif algorithm == 'powell':
            max_rslt = self._powell()

        # Finishing.
        return max_rslt

    def derived_attributes(self):
        """ Construct derived attributes.
        """
        # Antibugging
        assert (self.get_status() is True)

        # Distribute class attributes
        model_obj = self.get_attr('model_obj')
        paras_obj = self.get_attr('paras_obj')

        # Criterion function
        crit_func = CritCls(model_obj, paras_obj)

        crit_func.lock()

        self.attr['crit_func'] = crit_func

    ''' Private Methods.
    '''

    def _powell(self):
        """ Method that performs the Powell maximization.
        """
        # Antibugging
        assert (self.get_status() is True)

        # Distribute class attributes
        model_obj = self.get_attr('model_obj')
        paras_obj = self.get_attr('paras_obj')
        maxiter = model_obj.get_attr('maxiter')
        crit_func = self.get_attr('crit_func')

        # Staring values
        starting_values = paras_obj.get_values('external', 'free')

        rslt = fmin_powell(func=scipy_wrapper_function, x0=starting_values,
                           args=(crit_func,), xtol=0.0000000001,
                           ftol=0.0000000001, maxiter=maxiter, maxfun=None,
                           full_output=True, disp=1, callback=None)

        # Prepare result dictionary
        max_rslt = dict()

        max_rslt['xopt'] = np.array(rslt[0], ndmin=1)
        max_rslt['fun'] = rslt[1]
        max_rslt['grad'] = None
        max_rslt['success'] = (rslt[5] == 0)

        # Message
        max_rslt['message'] = rslt[5]

        if max_rslt['message'] == 1:
            max_rslt['message'] = 'Maximum number of function evaluations.'

        if max_rslt['message'] == 0:
            max_rslt['message'] = 'None'

        if max_rslt['message'] == 2:
            max_rslt['message'] = 'Maximum number of iterations.'

            # Finishing.
        return max_rslt

    def _bfgs(self):
        """ Method that performs a BFGS maximization.
        """
        # Antibugging
        assert (self.get_status() is True)

        # Distribute class attributes
        model_obj = self.get_attr('model_obj')
        paras_obj = self.get_attr('paras_obj')
        crit_func = self.get_attr('crit_func')
        maxiter = model_obj.get_attr('maxiter')
        gtol = model_obj.get_attr('gtol')
        epsilon = model_obj.get_attr('epsilon')

        # Staring values
        starting_values = paras_obj.get_values(version='external', which='free')

        # Maximization
        rslt = fmin_bfgs(f=scipy_wrapper_function,
                         fprime=scipy_wrapper_gradient, x0=starting_values,
                         args=(crit_func,), gtol=gtol, epsilon=epsilon,
                         maxiter=maxiter, full_output=True, disp=1, retall=0,
                         callback=None)

        # Prepare result dictionary
        max_rslt = dict()

        max_rslt['xopt'] = np.array(rslt[0], ndmin=1)
        max_rslt['fun'] = rslt[1]
        max_rslt['grad'] = rslt[2]
        max_rslt['covMat'] = rslt[3]
        max_rslt['success'] = (rslt[6] == 0)

        # Message
        max_rslt['message'] = rslt[6]

        if max_rslt['message'] == 1:
            max_rslt['message'] = 'Maximum number of function evaluations.'

        if max_rslt['message'] == 0:
            max_rslt['message'] = 'None'

        if max_rslt['message'] == 2:
            max_rslt['message'] = 'Gradient and/or function calls not changing.'

        # Finishing.
        return max_rslt
