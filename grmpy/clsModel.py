''' Module that contains the model object.
'''

# standard library.
import sys

import statsmodels.api  as sm
import numpy            as np

# project library
from grmpy.clsMeta import MetaCls


class modelCls(MetaCls):
    
    def __init__(self):
        
        self.attr = {}
        
        # Data matrices.
        self.attr['numAgents'] = None

        self.attr['Y'] = None
        
        self.attr['D'] = None
        
        self.attr['xExPost'] = None
        
        self.attr['xExAnte'] = None
        
        self.attr['G']       = None
        
        self.attr['Z']       = None


        self.attr['numCovarsExclBeneExPost'] = None

        self.attr['numCovarsExclBeneExAnte'] = None

        self.attr['numCovarsExclCost']       = None
        
        # Endogenous objects.
        self.attr['P']       = None
                

        self.attr['xExPostEval'] = None 

        self.attr['xExAnteEval'] = None 

        self.attr['zEval']       = None

        self.attr['cEval']       = None  

        
        self.attr['commonSupport']      = None


        self.attr['withoutPrediction']  = None
        
        self.attr['surpEstimation']     = None

        # Optional arguments.
        self.attr['algorithm']       = None
        self.attr['epsilon']         = None
        self.attr['differences']     = None

        self.attr['gtol']            = None
        self.attr['maxiter']         = None

        self.attr['withAsymptotics'] = None
        self.attr['numDraws']        = None

        self.attr['version']         = None
        self.attr['hessian']         = None
        self.attr['alpha']           = None


        # Status.               
        self.is_locked = False
    
    ''' Private class methods.
    '''    
    def derived_attributes(self):
        ''' Calculate derived attributes.
        '''
        # Number of agents.
        self.attr['numAgents']   = self.attr['xExPost'].shape[0]
        
        # Evaluation points.
        self.attr['xExPostEval'] = self.attr['xExPost'].mean(axis = 0)

        self.attr['xExAnteEval'] = self.attr['xExAnte'].mean(axis = 0)

        self.attr['zEval']       = self.attr['Z'].mean(axis = 0)

        self.attr['cEval']       = self.attr['G'].mean(axis = 0)
                
        # Common Support.
        self.attr['P'], self.attr['commonSupport'] = self._getCommonSupport()

        # Prediction.
        self.attr['withoutPrediction'] = \
            (self.attr['xExPost'].shape[1] == self.attr['xExAnte'].shape[1])
            
        # Surplus estimation.
        self.attr['surpEstimation'] = \
            (self.attr['numCovarsExclBeneExAnte'] > 0)
        
    def _getCommonSupport(self):
        ''' Calculate common support.
        '''
        # Antibugging.
        assert (self.get_status() == True)
        
        # Distribute attributes.
        D = self.get_attr('D')
        
        Z = self.get_attr('Z')
                
        # Probit estimation.            
        sys.stdout = open('/dev/null', 'w')
            
        rslt = sm.Probit(D, Z)
        
        P    = rslt.predict(rslt.fit().params)
            
        sys.stdout = sys.__stdout__
            
        # Determine common support.
        lowerBound = np.round(max(min(P[D == 1]), min(P[D == 0])), decimals = 2)
        
        upperBound = np.round(min(max(P[D == 1]), max(P[D == 0])), decimals = 2)
            
        # Finishing.
        return P, (lowerBound, upperBound)

    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''
        # Antibugging.
        assert (self.get_status() == True)
        
        # Outcome and treatment variable.
        for type_ in ['Y', 'D']:

            assert (isinstance(self.attr[type_], np.ndarray))
            assert (np.all(np.isfinite(self.attr[type_])))
            assert (self.attr[type_].dtype == 'float')
            assert (self.attr[type_].shape == (self.attr['numAgents'],))
        
        # Prediction step.
        assert (self.attr['withoutPrediction'] in [True, False])
        
        # Surplus estimation.
        assert (self.attr['surpEstimation'] in [True, False])
        
        # Number of agents.
        assert (isinstance(self.attr['numAgents'], int))
        assert (self.attr['numAgents'] > 0)
        
        # Class status.
        assert (self.is_locked in [True, False])
        
        # Covariate containers.
        for type_ in ['xExPost', 'xExAnte', 'G', 'Z']:

            if(self.attr[type_] is not None):

                assert (isinstance(self.attr[type_], np.ndarray))
                assert (np.all(np.isfinite(self.attr[type_])))
                assert (self.attr[type_].ndim == 2)

        # Propensity score.
        assert (isinstance(self.attr['P'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['P'])))
        assert (self.attr['P'].ndim == 1)

        # Counts.
        for type_ in ['numCovarsExclCost', 'numCovarsExclBeneExAnte']:
            
            assert (isinstance(self.attr[type_], int))
            assert (self.attr[type_] >= 0)
        
        # Evaluation points.
        for type_ in ['xExPostEval', 'xExAnteEval', 'zEval', 'cEval']:
        
            assert (isinstance(self.attr[type_], np.ndarray))
            assert (np.all(np.isfinite(self.attr[type_])))
            assert (self.attr[type_].ndim == 1)
        
        # Common support.
        assert (isinstance(self.attr['commonSupport'], tuple))
        assert (len(self.attr['commonSupport']) == 2)       

        # version
        assert (self.attr['version'] in ['fast', 'slow'])

        # withAsymptotics.
        assert (self.attr['withAsymptotics'] in [True, False])

        # Algorithm.
        assert (self.attr['algorithm'] in ['bfgs', 'powell'])

        # Maximum iteration.
        if(self.attr['maxiter'] is not None):

            assert (isinstance(self.attr['maxiter'], int))
            assert (self.attr['maxiter'] >= 0)

        # alpha.
        assert (isinstance(self.attr['alpha'], float))
        assert (0.0 < self.attr['alpha'] < 1.0)

        # gtol.
        assert (isinstance(self.attr['gtol'], float))
        assert (self.attr['gtol'] > 0.00)

        # epsilon.
        assert (isinstance(self.attr['epsilon'], float))
        assert (self.attr['epsilon'] > 0.00)

        # differences.
        assert (self.attr['differences'] in ['one-sided', 'two-sided'])

        # hessian.
        assert (self.attr['hessian'] in ['bfgs', 'numdiff'])

        if(self.attr['algorithm'] == 'powell'):

            assert(self.attr['hessian'] == 'numdiff')