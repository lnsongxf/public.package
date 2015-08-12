""" Module that contains the model object.
"""

# standard library.
import sys

import statsmodels.api as sm
import numpy as np


# project library
from grmpy.clsMeta import MetaCls


class ModelCls(MetaCls):
    def __init__(self):

        self.attr = dict()

        # Data matrices
        self.attr['num_agents'] = None
        self.attr['Y'] = None
        self.attr['D'] = None
        self.attr['X_ex_post'] = None
        self.attr['X_ex_ante'] = None
        self.attr['G'] = None
        self.attr['Z'] = None
        self.attr['num_covars_excl_bene_ex_post'] = None
        self.attr['num_covars_excl_bene_ex_ante'] = None
        self.attr['num_covars_excl_cost'] = None

        # Endogenous objects
        self.attr['P'] = None
        self.attr['x_ex_post_eval'] = None
        self.attr['x_ex_ante_eval'] = None
        self.attr['z_eval'] = None
        self.attr['c_eval'] = None
        self.attr['common_support'] = None
        self.attr['without_prediction'] = None
        self.attr['surp_estimation'] = None

        # Optional arguments
        self.attr['algorithm'] = None
        self.attr['epsilon'] = None
        self.attr['differences'] = None
        self.attr['gtol'] = None
        self.attr['maxiter'] = None
        self.attr['with_asymptotics'] = None
        self.attr['num_draws'] = None
        self.attr['version'] = None
        self.attr['hessian'] = None
        self.attr['alpha'] = None

        # Status
        self.is_locked = False

    """ Private class methods.
    """

    def derived_attributes(self):
        """ Calculate derived attributes.
        """
        # Number of agents 
        self.attr['num_agents'] = self.attr['X_ex_post'].shape[0]

        # Evaluation points 
        self.attr['x_ex_post_eval'] = self.attr['X_ex_post'].mean(axis=0)
        self.attr['x_ex_ante_eval'] = self.attr['X_ex_ante'].mean(axis=0)
        self.attr['z_eval'] = self.attr['Z'].mean(axis=0)
        self.attr['c_eval'] = self.attr['G'].mean(axis=0)

        # Common Support 
        self.attr['P'], self.attr['common_support'] = self._get_common_support()

        # Prediction 
        self.attr['without_prediction'] = \
            (self.attr['X_ex_post'].shape[1] == self.attr['X_ex_ante'].shape[1])

        # Surplus estimation 
        self.attr['surp_estimation'] = \
            (self.attr['num_covars_excl_bene_ex_ante'] > 0)

    def _get_common_support(self):
        """ Calculate common support.
        """
        # Antibugging.
        assert (self.get_status() is True)

        # Distribute attributes
        d = self.get_attr('D')
        z = self.get_attr('Z')

        # Probit estimation
        stdout_current = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        rslt = sm.Probit(d, z)
        p = rslt.predict(rslt.fit().params)
        sys.stdout = stdout_current

        # Determine common support
        lower_bound = np.round(max(min(p[d == 1]), min(p[d == 0])), decimals=2)
        upper_bound = np.round(min(max(p[d == 1]), max(p[d == 0])), decimals=2)

        # Finishing
        return p, (lower_bound, upper_bound)

    def _check_integrity(self):
        """ Check integrity of class instance.
        """
        # Antibugging
        assert (self.get_status() is True)

        # Outcome and treatment variable
        for type_ in ['Y', 'D']:
            assert (isinstance(self.attr[type_], np.ndarray))
            assert (np.all(np.isfinite(self.attr[type_])))
            assert (self.attr[type_].dtype == 'float')
            assert (self.attr[type_].shape == (self.attr['num_agents'],))

        # Prediction step
        assert (self.attr['without_prediction'] in [True, False])

        # Surplus estimation
        assert (self.attr['surp_estimation'] in [True, False])

        # Number of agents
        assert (isinstance(self.attr['num_agents'], int))
        assert (self.attr['num_agents'] > 0)

        # Class status
        assert (self.is_locked in [True, False])

        # Covariate containers
        for type_ in ['X_ex_post', 'X_ex_ante', 'G', 'Z']:
            if self.attr[type_] is not None:
                assert (isinstance(self.attr[type_], np.ndarray))
                assert (np.all(np.isfinite(self.attr[type_])))
                assert (self.attr[type_].ndim == 2)

        # Propensity score
        assert (isinstance(self.attr['P'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['P'])))
        assert (self.attr['P'].ndim == 1)

        # Counts 
        for type_ in ['num_covars_excl_cost', 'num_covars_excl_bene_ex_ante']:
            assert (isinstance(self.attr[type_], int))
            assert (self.attr[type_] >= 0)

        # Evaluation points 
        for type_ in ['x_ex_post_eval', 'x_ex_ante_eval', 'z_eval', 'c_eval']:
            assert (isinstance(self.attr[type_], np.ndarray))
            assert (np.all(np.isfinite(self.attr[type_])))
            assert (self.attr[type_].ndim == 1)

        # Common support 
        assert (isinstance(self.attr['common_support'], tuple))
        assert (len(self.attr['common_support']) == 2)

        # version
        assert (self.attr['version'] in ['fast', 'slow'])

        # with_asymptotics 
        assert (self.attr['with_asymptotics'] in [True, False])

        # Algorithm
        assert (self.attr['algorithm'] in ['bfgs', 'powell'])

        # Maximum iteration 
        if self.attr['maxiter'] is not None:
            assert (isinstance(self.attr['maxiter'], int))
            assert (self.attr['maxiter'] >= 0)

        # alpha
        assert (isinstance(self.attr['alpha'], float))
        assert (0.0 < self.attr['alpha'] < 1.0)

        # gtol
        assert (isinstance(self.attr['gtol'], float))
        assert (self.attr['gtol'] > 0.00)

        # epsilon
        assert (isinstance(self.attr['epsilon'], float))
        assert (self.attr['epsilon'] > 0.00)

        # differences
        assert (self.attr['differences'] in ['one-sided', 'two-sided'])

        # hessian
        assert (self.attr['hessian'] in ['bfgs', 'numdiff'])

        if self.attr['algorithm'] == 'powell':
            assert (self.attr['hessian'] == 'numdiff')
