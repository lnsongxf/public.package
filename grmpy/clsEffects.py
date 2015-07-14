''' Module that holds the effects class.  
'''
# standard library
import numpy as np

import scipy
import random

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

class effectCls(metaCls):
    ''' Lightweight class for the simulation of treatment effect parameters.
    '''
    def __init__(self):
        
        self.isLocked = False
    
    def getEffects(self, modelObj, parasObj, type_, args):
        ''' Get effects.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        assert (isinstance(modelObj, modelCls))
        assert (modelObj.getStatus() == True)
        
        assert (isinstance(parasObj, parasCls))
        assert (parasObj.getStatus() == True)

        rslt = self.getTreatmentParameter(modelObj, parasObj, args['which'])
        
        # Finishing.
        return rslt
    
    ''' Private  methods.
    '''
    def getTreatmentParameter(self, modelObj, parasObj, which):
        ''' Construct marginal effect parameters.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (which in ['smteExAnte', 'cmteExAnte', 'bmteExPost'])
        
        assert (isinstance(modelObj, modelCls))
        assert (modelObj.getStatus() == True)
        
        assert (isinstance(parasObj, parasCls))
        assert (parasObj.getStatus() == True)
        
        # Distribute class attributes.
        xExPostEval = modelObj.getAttr('xExPostEval')                
        zEval       = modelObj.getAttr('zEval')
        cEval       = modelObj.getAttr('cEval')
                
        # Marginal benefit of treatment.
        rhoU1V = parasObj.getParameters('rho', 'U1,V')
        rhoU0V = parasObj.getParameters('rho', 'U0,V')
        
        coeffsBeneExPost = parasObj.getParameters('bene', 'exPost')     
        coeffsCost       = parasObj.getParameters('cost', None) 
        coeffsChoc       = parasObj.getParameters('choice', None) 
                  
        sdV  = parasObj.getParameters('sd', 'V')        
        sdU1 = parasObj.getParameters('sd', 'U1')            
        sdU0 = parasObj.getParameters('sd', 'U0')    

        bmteLevel = np.dot(coeffsBeneExPost, xExPostEval)
        smteLevel = np.dot(coeffsChoc, zEval)
        cmteLevel = np.dot(coeffsCost, cEval)
        
        bmteExPost = np.tile(np.nan, (99))
        cmteExAnte = np.tile(np.nan, (99))
        smteExAnte = np.tile(np.nan, (99))
                
        evalPoints = np.round(np.arange(0.01, 1.0, 0.01), decimals = 2)
        quantiles  = scipy.stats.norm.ppf(evalPoints, loc = 0, scale = sdV)
        
        # Construct marginal benefit of treatment (ex post)
        bmteSlopes = ((sdU1/sdV)*rhoU1V - (sdU0/sdV)*rhoU0V)*quantiles
        smteSlopes = -quantiles
        cmteSlopes = (((sdU1/sdV)*rhoU1V - (sdU0/sdV)*rhoU0V) + 1.0)*quantiles

        # Construct marginal benefit of treatment (ex post)
        for i in range(99):
            
            bmteExPost[i] = bmteLevel + bmteSlopes[i]
                    
        # Construct marginal surplus of treatment (ex ante).        
        for i in range(99):
                
            smteExAnte[i] = smteLevel + smteSlopes[i]
    
        # Construct marginal cost of treatment (ex ante).
        for i in range(99):
                
            cmteExAnte[i] = cmteLevel + cmteSlopes[i]
            
        if(which == 'bmteExPost'):
        
            rslt = bmteExPost
        
        elif(which == 'cmteExAnte'):
            
            rslt = cmteExAnte
            
        elif(which == 'smteExAnte'):
            
            rslt = smteExAnte
    
        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        assert (rslt.shape == (99, ))
    
        # Finishing.    
        return rslt

    ''' Auxiliary methods.
    '''
    def _getMarginalEffects(self, evalPoint, xExPostEval, xExAnteEval, 
                           zEval, cEval, parasObj):
        ''' Get the marginal effect parameters for a particular point of 
            evaluation and location. 
        '''    
        # Antibugging.
        assert (self.getStatus() == True)
        
        assert (isinstance(evalPoint, float))
        assert (0.0 < evalPoint < 1.00)
        
        assert (isinstance(xExPostEval, np.ndarray))
        assert (np.all(np.isfinite(xExPostEval)))
        
        assert (isinstance(xExAnteEval, np.ndarray))
        assert (np.all(np.isfinite(xExAnteEval)))
        
        assert (isinstance(zEval, np.ndarray))
        assert (np.all(np.isfinite(zEval)))
    
        assert (isinstance(cEval, np.ndarray))
        assert (np.all(np.isfinite(cEval)))    
        
        # Distribute class attributes.
        rhoU1V = parasObj.getParameters('rho', 'U1,V')
        rhoU0V = parasObj.getParameters('rho', 'U0,V')
                
        coeffsBeneExPost = parasObj.getParameters('bene', 'exPost')     
        coeffsBeneExAnte = parasObj.getParameters('bene', 'exAnte')    
        coeffsChoc       = parasObj.getParameters('choice', None) 
        coeffsCost       = parasObj.getParameters('cost', None) 
                
        sdV  = parasObj.getParameters('sd', 'V')        
        sdU1 = parasObj.getParameters('sd', 'U1')            
        sdU0 = parasObj.getParameters('sd', 'U0')    
        
        bmteExPostLevel = np.dot(coeffsBeneExPost, xExPostEval)
        bmteExAnteLevel = np.dot(coeffsBeneExAnte, xExAnteEval)
        
        smteLevel = np.dot(coeffsChoc, zEval)
        cmteLevel = np.dot(coeffsCost, cEval)
                    
        quantile  = scipy.stats.norm.ppf(evalPoint, loc = 0, scale = sdV)
                
        # Construct marginal benefit of treatment (ex post)
        bmteSlope = ((sdU1/sdV)*rhoU1V - (sdU0/sdV)*rhoU0V)*quantile
        smteSlope = -quantile
        cmteSlope = (((sdU1/sdV)*rhoU1V - (sdU0/sdV)*rhoU0V) + 1.0)*quantile
    
        # Construct marginal effects of treatment.
        bmteExPost = bmteExPostLevel + bmteSlope
        bmteExAnte = bmteExAnteLevel + bmteSlope    
    
        smteExAnte = smteLevel + smteSlope 
        cmteExAnte = cmteLevel + cmteSlope
    
        # Quality checks.
        for obj in [bmteExPost, bmteExAnte, cmteExAnte, smteExAnte]:
            
            assert (isinstance(obj, float))
            assert (np.isfinite(obj))
        
        assert (np.round(smteExAnte - (bmteExAnte - cmteExAnte), \
                    decimals = 10) == 0.0)
            
        # Finishing.    
        return bmteExPost, bmteExAnte, cmteExAnte, smteExAnte