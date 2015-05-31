''' Module that contains the model object.
'''

# standard library.
import sys

import statsmodels.api  as sm
import numpy            as np

# project library
from grmpy.clsMeta import metaCls


class modelCls(metaCls):
    
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
        
        # Status.               
        self.isLocked = False
    
    ''' Private class methods.
    '''    
    def _derivedAttributes(self):
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
        assert (self.getStatus() == True)
        
        # Distribute attributes.
        D = self.getAttr('D') 
        
        Z = self.getAttr('Z')
                
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
        assert (self.getStatus() == True)
        
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
        assert (self.isLocked in [True, False])
        
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
