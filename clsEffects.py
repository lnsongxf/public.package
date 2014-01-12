''' Module that holds the effects class.  
'''
# standard library
import numpy as np

import scipy
import random

# project library
import clsMeta
import clsModel
import clsParas

class effectCls(clsMeta.meta):
    ''' Lightweight class for the simulation of treatment effect parameters.
    '''
    def __init__(self):
        
        self.isLocked = False
    
    def getEffects(self, modelObj, parasObj, type_, args):
        ''' Get effects.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        assert (isinstance(modelObj, clsModel.modelCls)) 
        assert (modelObj.getStatus() == True)
        
        assert (isinstance(parasObj, clsParas.parasCls)) 
        assert (parasObj.getStatus() == True)

        assert (type_ in ['marginal', 'average'])

        if(type_ in ['marginal']):
            
            assert (set(args.keys()) == set(['which']))
        
        # Get effects.
        if(type_ == 'marginal'):
            
            which = args['which']
 
            rslt  = self.getTreatmentParameter(modelObj, parasObj, which)
        
        if(type_ == 'average'):
            
            isConditional = args['isConditional']
        
            numSims       = args['numSims']
            
            rslt = self.getAverageEffects(modelObj, parasObj, numSims, isConditional)
        
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
        
        assert (isinstance(modelObj, clsModel.modelCls)) 
        assert (modelObj.getStatus() == True)
        
        assert (isinstance(parasObj, clsParas.parasCls)) 
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
    
    def getAverageEffects(self, modelObj, parasObj, numSims, isConditional):
        ''' Simulation of average effects of treatment.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isConditional in [True, False])
        assert (isinstance(numSims, int))
        assert (numSims > 0)
        
        assert (isinstance(modelObj, clsModel.modelCls)) 
        assert (modelObj.getStatus() == True)
        
        assert (isinstance(parasObj, clsParas.parasCls)) 
        assert (parasObj.getStatus() == True)

        # Distribute class attributes.
        numAgents      = modelObj.getAttr('numAgents')
        commonSupport  = modelObj.getAttr('commonSupport')
        
        # Initialize result dictionary.
        rslt = {}
        
        rslt['bteExPost'] = {}
        rslt['bteExPost']['average']    = np.empty(0)
        rslt['bteExPost']['treated']    = np.empty(0)
        rslt['bteExPost']['untreated']  = np.empty(0)
        
        rslt['bteExAnte'] = {}
        rslt['bteExAnte']['average']    = np.empty(0)
        rslt['bteExAnte']['treated']    = np.empty(0)
        rslt['bteExAnte']['untreated']  = np.empty(0)
        
        rslt['cte'] = {}
        rslt['cte']['average']          = np.empty(0)
        rslt['cte']['treated']          = np.empty(0)
        rslt['cte']['untreated']        = np.empty(0)
        
        rslt['ste'] = {}
        rslt['ste']['average']          = np.empty(0)
        rslt['ste']['treated']          = np.empty(0)
        rslt['ste']['untreated']        = np.empty(0)
        
        # Construct bounds.
        lowerBound = 0.01
        upperBound = 1.00
        
        if(isConditional):
            
            lowerBound = max(commonSupport[0], 0.01)
            
            upperBound = min(commonSupport[1], 0.99)

        # Determine choice.
        D = np.empty(0)
            
        bmteExPostAll = np.empty(0)
        bmteExAnteAll = np.empty(0)
            
        cmteExAnteAll = np.empty(0)
        smteExAnteAll = np.empty(0)
            
        # Main loop.
        for _ in range(numSims):
                
            # Draw identifier and evaluation point.
            idx = np.random.randint(0, numAgents)
            u   = random.choice(np.round(np.arange(lowerBound, upperBound, \
                    0.01), decimals = 2))
            
            # Select evaluation points.
            xExPostEval = modelObj.getAttr('xExPost')[idx, :]
            xExAnteEval = modelObj.getAttr('xExAnte')[idx, :]
                
            cEval = modelObj.getAttr('G')[idx, :]
            zEval = modelObj.getAttr('Z')[idx, :]
                
            # Calculate choice.
            coeffsChoc = parasObj.getParameters('choice', None) 
            sdV        = parasObj.getParameters('sd', 'V')
            V          = scipy.stats.norm.ppf(u, loc = 0, scale = sdV)
                
            D = np.append(D, (np.dot(coeffsChoc, zEval) - V > 0.0))
            
            # Construct.
            bmteExPost, bmteExAnte, cmteExAnte, smteExAnte = \
                self._getMarginalEffects(u, xExPostEval, xExAnteEval, zEval, cEval, parasObj)
               
            # Collect.
            bmteExPostAll = np.append(bmteExPostAll, bmteExPost)
            bmteExAnteAll = np.append(bmteExAnteAll, bmteExAnte)    
            
            cmteExAnteAll = np.append(cmteExAnteAll, cmteExAnte)    
            smteExAnteAll = np.append(smteExAnteAll, smteExAnte)    
            
        # Finishing.
        for obj in [bmteExPostAll, bmteExAnteAll, cmteExAnteAll, smteExAnteAll]:
       
            assert (isinstance(obj, np.ndarray))
            assert (np.all(np.isfinite(obj)))
            assert (obj.shape == (numSims,))
            assert (obj.dtype == 'float')
            
        # Collect results.
        rslt['bteExPost']['average']   = bmteExPostAll.mean()
        rslt['bteExPost']['treated']   = bmteExPostAll[D == True].mean()
        rslt['bteExPost']['untreated'] = bmteExPostAll[D == False].mean()
            
        rslt['bteExAnte']['average']   = bmteExAnteAll.mean()
        rslt['bteExAnte']['treated']   = bmteExAnteAll[D == True].mean()
        rslt['bteExAnte']['untreated'] = bmteExAnteAll[D == False].mean()
            
        rslt['cte']['average']   = cmteExAnteAll.mean()
        rslt['cte']['treated']   = cmteExAnteAll[D == True].mean()
        rslt['cte']['untreated'] = cmteExAnteAll[D == False].mean()
            
        rslt['ste']['average']   = smteExAnteAll.mean()
        rslt['ste']['treated']   = smteExAnteAll[D == True].mean()
        rslt['ste']['untreated'] = smteExAnteAll[D == False].mean()
        
        # Antibugging.
        for obj in ['bteExPost', 'bteExAnte', 'cte', 'ste']:
            
            for subgroup in ['average', 'treated', 'untreated']:
                
                assert (np.isfinite(rslt[obj][subgroup]))
        
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