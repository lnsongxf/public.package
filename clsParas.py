''' Module that holds the parasCls, which manages all things related to the
    parameter management.
'''
# standard library
import statsmodels.api as sm
import numpy           as np

import sys
import scipy
import random

class parasCls(object):
    ''' Class for the parameter management.
    '''
    def __init__(self):
        
        # Attach attributes.
        self.attr = {}

        self.attr['numAgents']  = None        
        self.attr['paraObjs']   = []
        self.attr['numParas']   = 0
        self.attr['numFree']    = 0

        self.attr['xExAnte']    = None
        self.attr['xExPost']    = None
        self.attr['D']          = None
        self.attr['G']          = None
        self.attr['Z']          = None
                
        self.attr['factor']     = None
        self.attr['numSims']    = None


        self.attr['numCovarsExclBeneExAnte'] = None            
        self.attr['numCovarsExclBeneExPost'] = None
        
        self.attr['numCovarsBeneExPost']     = None
        self.attr['numCovarsBeneExAnte']     = None
        
        self.attr['numCovarsExclCost']       = None
        
        self.attr['numCovarsOutc']           = None
        self.attr['numCovarsChoc']           = None
        self.attr['numCovarsCost']           = None

        self.attr['numCoeffsBene']           = None

        self.attr['commonSupport']           = None
        self.attr['normalization']           = None
        self.attr['isNormalized']            = None
        self.attr['withoutPrediction']       = None
                                         
        # Evaluation points of parameters.
        self.attr['xExPostEval']             = None
        self.attr['xExAnteEval']             = None
        self.attr['cEval']                   = None
        self.attr['zEval']                   = None
                                                        
        # Status.
        self.isLocked = False
    
    def addParameter(self, type_, subgroup, value, isFree, bounds):
        ''' Add parameters to class instance.
        '''
        # Antibugging.
        assert (isFree in  [True, False])
        assert (self.getStatus() == False)
        assert (len(bounds) == 2)

        if(type_ == 'sd'):
            
            assert (bounds[0] is not None)
            assert (bounds[0] > 0.0)
            
        if(type_ == 'rho'):
            
            assert (bounds[0] > -1.00)
            assert (bounds[1] <  1.00)

        # Initialize parameters.
        paraObj = _paraContainer()
        
        paraObj.setAttr('type', type_)
        
        paraObj.setAttr('subgroup', subgroup)
        
        paraObj.setAttr('value', value)
        
        paraObj.setAttr('isFree', isFree)

        paraObj.setAttr('startVal', value)

        paraObj.setAttr('bounds', bounds)
                     
        paraObj.lock()
                                 
        # Update class attributes.
        self.attr['paraObjs'].append(paraObj)

        self.attr['numParas'] += 1

        if(isFree):
            
            self.attr['numFree'] += 1

    def getParameter(self, count):
        ''' Get a single parameter object identified by its count. It is 
            important to note, that the selection mechanism refers to all
            parameters not just the true ones. 
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (count < self.getAttr('numParas'))
        
        # Algorithm.
        rslt = self.getAttr('paraObjs')[count]

        # Finishing.
        return rslt

    def getParameters(self, type_, subgroup, isObj = False):
        ''' Get parameter groups.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkRequest(type_, subgroup, isObj) == True)
            
        # Collect request.
        rsltList = []
            
        if(type_ in ['outc', 'cost', 'rho', 'sd']):
            
            for paraObj in self.attr['paraObjs']:
                
                if(paraObj.getAttr('type')      != type_):    continue
                
                if(paraObj.getAttr('subgroup')  != subgroup): continue
                
                if(isObj):
                    
                    rsltList.append(paraObj)
        
                else:
                    
                    rsltList.append(paraObj.getAttr('value'))
        
        # Special types: Covariances.
        if(type_ == 'cov'):
            
            varOne = subgroup.split(',')[0]
            varTwo = subgroup.split(',')[1]
            
            rho    = self.getParameters('rho', varOne + ',' + varTwo)
            sdOne  = self.getParameters('sd', varOne)
            sdTwo  = self.getParameters('sd', varTwo)
            
            rsltList.append(rho*sdOne*sdTwo)
        
        # Special types: Variances.
        if(type_ == 'var'):
            
            sdOne  = self.getParameters('sd', subgroup)
            
            rsltList.append(sdOne**2)
        
        if(type_ == 'bene'):
            
            assert (subgroup in ['exPost', 'exAnte'])
            
            if(subgroup == 'exPost'):

                outcTreated   = self.getParameters('outc', 'treated')
                outcUntreated = self.getParameters('outc', 'untreated')
        
                rsltList = outcTreated - outcUntreated
        
            elif(subgroup == 'exAnte'):
                
                coeffsBeneExAnte = self._predictionStep()
                
                rsltList = coeffsBeneExAnte
        
        if(type_ == 'choice'):
            
            numCovarExclCost        = self.getAttr('numCovarsExclCost')
            numCovarExclBeneExAnte  = self.getAttr('numCovarsExclBeneExAnte')
            
            coeffsBeneExAnte = self.getParameters('bene', 'exAnte')   
            coeffsCost       = self.getParameters('cost', None)   

            coeffsBene = np.concatenate((coeffsBeneExAnte, np.tile(0.0, numCovarExclCost)))
            coeffsCost = np.concatenate((np.tile(0.0, numCovarExclBeneExAnte), coeffsCost))

            rsltList = coeffsBene - coeffsCost
        
        # Dealing with objects.
        if(isObj and type_ in ['rho', 'sd', 'var', 'cov']):
        
            return rsltList[0]
        
        # Type conversion.
        rslt = np.array(rsltList[:])
                    
        if(type_ in ['rho', 'sd', 'var', 'cov']):
    
            rslt = np.array(rslt[0])
                
        # Quality check.
        if(isObj == False):

            assert (isinstance(rslt, np.ndarray))
            assert (np.all(np.isfinite(rslt)))
            
        # Finishing.
        return rslt

    ''' Public get/set methods.
    '''
    def setAttr(self, key, arg):
        ''' Set attribute.
        '''
        # Antibugging.
        assert (self.getStatus() == False)
        assert (self._checkKey(key) == True)
        
        # Set attribute.
        self.attr[key] = arg

    def getAttr(self, key):
        ''' Get attribute.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkKey(key) == True)
        
        # Select from dictionary.
        return self.attr[key]      

    def getStatus(self):
        ''' Get status of class instance.
        '''
                
        # Finishing.
        return self.isLocked
 
    def unlock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == True) 
        
        # Update status.
        self.isLocked = False       
            
    def lock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == False)
        
        # Finishing.        
        self.isLocked = True
        
        # Endogenous attributes.
        self._finalizeConstruction()

        # Check quality.
        assert (self._checkIntegrity() == True)
        assert (self.getStatus() == True)
            
    def getValues(self, isExternal = False):
        ''' Get all free parameter values.
        '''    
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkIntegrity() == True)
        assert (isExternal in [True, False])

        # Main algorithm.
        paraObjs = self.attr['paraObjs']
        
        rslt = []
        
        for paraObj in paraObjs:
            
            if(paraObj.getAttr('isFree') is False): continue
            
            value = paraObj.getAttr('value')
            
            if(isExternal):

                value = self._transformToExternal(paraObj, paraObj.getAttr('value'))
                
            rslt.append(value)
            
        # Type conversion.
        rslt = np.array(rslt)
        
        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        assert (rslt.shape == (self.attr['numFree'], ))
        
        # Finishing.
        return rslt
   
    ''' All methods related to construction of the objects.
    '''
    def getAverageEffects(self, isConditional = False):
        ''' Simulation of average effects of treatment.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isConditional in [True, False])

        # Distribute class attributes.
        numAgents      = self.getAttr('numAgents')
        numSims        = self.getAttr('numSims')
        commonSupport  = self.getAttr('commonSupport')
        
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
            
            lowerBound, upperBound = commonSupport[0], commonSupport[1]  
        
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
            xExPostEval = self.getAttr('xExPost')[idx, :]
            xExAnteEval = self.getAttr('xExAnte')[idx, :]
                
            cEval = self.getAttr('G')[idx, :]
            zEval = self.getAttr('Z')[idx, :]
                
            # Calculate choice.
            coeffsChoc = self.getParameters('choice', None) 
            sdV        = self.getParameters('sd', 'V')
            V          = scipy.stats.norm.ppf(u, loc = 0, scale = sdV)
                
            D = np.append(D, (np.dot(coeffsChoc, zEval) - V > 0.0))
            
            # Construct.
            bmteExPost, bmteExAnte, cmteExAnte, smteExAnte = \
                self.getMarginalEffects(u, xExPostEval, xExAnteEval, zEval, cEval)
               
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
            
        # Finishing.
        return rslt
         
    def getMarginalEffects(self, evalPoint, xExPostEval, xExAnteEval, 
                           zEval, cEval):
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
        rhoU1V = self.getParameters('rho', 'U1,V')
        rhoU0V = self.getParameters('rho', 'U0,V')
                
        coeffsBeneExPost = self.getParameters('bene', 'exPost')     
        coeffsBeneExAnte = self.getParameters('bene', 'exAnte')    
        coeffsChoc       = self.getParameters('choice', None) 
        coeffsCost       = self.getParameters('cost', None) 
                
        sdV  = self.getParameters('sd', 'V')        
        sdU1 = self.getParameters('sd', 'U1')            
        sdU0 = self.getParameters('sd', 'U0')    
        
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
    
    def getTreatmentParameter(self, which):
        ''' Construct parameters.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (which in ['smteExAnte', 'cmteExAnte', 'bmteExPost'])
        
        # Marginal benefit of treatment.
        rhoU1V = self.getParameters('rho', 'U1,V')
        rhoU0V = self.getParameters('rho', 'U0,V')
        
        coeffsBeneExPost = self.getParameters('bene', 'exPost')     
        coeffsCost       = self.getParameters('cost', None) 
        coeffsChoc       = self.getParameters('choice', None) 
  
        xExPostEval = self.getAttr('xExPostEval')                
        zEval       = self.getAttr('zEval')
        cEval       = self.getAttr('cEval')
                
        sdV  = self.getParameters('sd', 'V')        
        sdU1 = self.getParameters('sd', 'U1')            
        sdU0 = self.getParameters('sd', 'U0')    

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

    ''' All methods related to updating the parameters. 
    '''     
    def updateValues(self, x):
        ''' Update all free parameters.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkIntegrity() == True)
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        assert (x.shape == (self.getAttr('numFree'), ))
        
        # Distribute class attributes.
        paraObjs = self.getAttr('paraObjs')
        
        counter = 0
        
        for paraObj in paraObjs:
           
            if(paraObj.getAttr('isFree') is False): continue
            
            value = x[counter]
 
            if(paraObj.getAttr('hasBounds')):
                
                value = self._transformToInternal(paraObj, value)
     
            paraObj.setAttr('value', value)
            
            counter += 1
        
        # Finishing.
        return True
    
    def _transformToExternal(self, paraObj, internalValue):
        ''' Transform internal values for external use by maximization 
            routine.
        '''
        # Antibugging.
        assert (isinstance(paraObj, _paraContainer))
        assert (paraObj.getStatus() == True)
        assert (isinstance(internalValue, float))
        assert (np.isfinite(internalValue))
        
        # Auxiliary objects.
        lowerBound, upperBound = paraObj.getAttr('bounds')
        
        hasLowerBound = (lowerBound is not None)
        hasUpperBound = (upperBound is not None)
        
        # Stabilization
        internalValue = self._clipInternalValue(paraObj, internalValue)
        
        # Upper bound only.
        if((not hasLowerBound) and (hasUpperBound)):
            
            externalValue = np.log(upperBound - internalValue)
        
        # Lower bound only.
        elif((hasLowerBound) and (not hasUpperBound)):
            
            externalValue = np.log(internalValue - lowerBound)
        
        # Upper and lower bounds.
        elif(hasLowerBound and hasUpperBound):
            
            interval  = upperBound - lowerBound
            transform = (internalValue - lowerBound)/interval
            
            externalValue = np.log(transform/(1.0 - transform))
        
        # No bounds.
        else:
            
            externalValue = internalValue
            
        # Quality Check.
        assert (isinstance(externalValue, float))
        assert (np.isfinite(externalValue))
        
        # Finishing.
        return externalValue
    
    def _transformToInternal(self, paraObj, externalValue):
        ''' Transform external values to internal paraObj.
        '''
        # Antibugging.
        assert (isinstance(paraObj, _paraContainer))
        assert (paraObj.getStatus() == True)
        assert (isinstance(externalValue, float))
        assert (np.isfinite(externalValue))
        
        # Auxiliary objects.
        lowerBound, upperBound = paraObj.getAttr('bounds') 
        
        hasBounds = paraObj.getAttr('hasBounds')
        
        hasLowerBound = (lowerBound is not None)
        hasUpperBound = (upperBound is not None)
        
        # Stabilization.        
        if(hasBounds):
            
            externalValue = np.clip(externalValue, None, 10)
    
        # Upper bound only.
        if((not hasLowerBound) and (hasUpperBound)):
            
            internalValue = upperBound - np.exp(externalValue)
                    
        # Lower bound only.
        elif((hasLowerBound) and (not hasUpperBound)):
            
            internalValue = lowerBound + np.exp(externalValue)
      
        # Upper and lower bounds.
        elif(hasLowerBound and hasUpperBound):
            
            interval      = upperBound - lowerBound
            internalValue = lowerBound + interval/(1.0 + np.exp(-externalValue)) 

        # No bounds.
        else:
            
            internalValue = externalValue

        # Stabilization.
        internalValue = self._clipInternalValue(paraObj, internalValue)

        # Quality Check.
        assert (isinstance(internalValue, float))
        assert (np.isfinite(internalValue))
        
        # Finishing.
        return internalValue
     
    def _clipInternalValue(self, paraObj, internalValue):
        ''' Assure that internal value not exactly equal to bounds.
        '''
        # Antibugging.
        assert (isinstance(paraObj, _paraContainer))
        assert (paraObj.getStatus() == True)
        assert (isinstance(internalValue, float))
        assert (np.isfinite(internalValue))
                
        # Auxiliary objects.
        lowerBound, upperBound = paraObj.getAttr('bounds')

        hasLowerBound = (lowerBound is not None)
        hasUpperBound = (upperBound is not None)
        
        # Check bounds.
        if(hasLowerBound):
            
            if(internalValue == lowerBound): internalValue += 0.01

        if(hasUpperBound):
            
            if(internalValue == upperBound): internalValue -= 0.01

        # Quality Check.
        assert (isinstance(internalValue, float))
        assert (np.isfinite(internalValue))
        
        if(hasLowerBound): assert (lowerBound < internalValue) 
        if(hasUpperBound): assert (upperBound > internalValue) 
            
        # Finishing.
        return internalValue
    
    ''' Additional private methods.
    '''
    def _finalizeConstruction(self):
        ''' Determine endogenous attributes.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Number of Covariates in Outcome.
        self.attr['numCovarsOutc'] = len(self.getParameters('bene', 'exPost'))
        
        # Number of Covariates in Outcome (ex ante).
        self.attr['numCovarsBeneExAnte'] = self.attr['xExAnte'].shape[1]

        # Number of Covariates in (ex post).
        self.attr['numCovarsBeneExPost'] = self.attr['xExPost'].shape[1]
        
        # Number of Covariates in Choice.
        self.attr['numCovarsChoc'] = len(self.getParameters('choice', None))

        # Number of Covariates in Cost.
        self.attr['numCovarsCost'] = len(self.getParameters('cost', None))
        
        # Number of Coefficients in Outcome.
        self.attr['numCoeffsBene'] = self.attr['numCovarsOutc']*2

        # Common Support.
        self.attr['commonSupport'] = self._getSupportRange()

    def _predictionStep(self):
        ''' Prediction step to account for benefit shifters unknown to the agent
            at the time of treatment decision. 
        '''
        # Antibugging.
        assert (self.getStatus() == True)

        # Distribute class attributes.
        withoutPrediction = self.getAttr('withoutPrediction')
        coeffsBeneExPost  = self.getParameters('bene', 'exPost')
        
        # Check applicability.
        if(withoutPrediction): return coeffsBeneExPost 
        
        xExPost = self.getAttr('xExPost')
        xExAnte = self.getAttr('xExAnte')               
    
        # Construct index.
        idxBene = np.dot(coeffsBeneExPost, xExPost.T)
        
        if(self.attr['factor'] is None):
     
            pinv = np.linalg.pinv(np.dot(xExAnte.T, xExAnte))
            
            self.attr['factor'] = np.dot(pinv, xExAnte.T)
                
        rslt = np.dot(self.attr['factor'], idxBene)
  
        # Type conversion.
        rslt = np.array(rslt)
        
        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        assert (rslt.shape == (self.getAttr('numCovarsBeneExAnte'), ))
        
        # Finishing.
        return rslt

    ''' Check integrity of class instance and attribute requests.
    '''
    def _checkRequest(self, type_, subgroup, obj):
        ''' Check the validity of the parameter request.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
     
        # Check type.
        assert (type_ in ['outc', 'cost', 'rho', 'sd', \
                          'var', 'cov', 'bene', 'choice'])
        
        # Check object.
        assert (obj in [True, False])

        if(obj): 
            
            assert (type_ in ['outc', 'cost', 'rho', 'sd']) 

        # Check subgroup.
        if(subgroup is not None): 
            
            assert (isinstance(subgroup, str))
        
        # Finishing.
        return True
    
    def _checkKey(self, key):
        ''' Check that key is present.
        '''        
        # Check presence.
        assert (key in self.attr.keys())
        
        # Finishing.
        return True
    
    def _getSupportRange(self):
        ''' Calculate common support.
        '''       
        # Distribute attributes.
        D           = self.getAttr('D') 
        Z           = self.getAttr('Z')
            
        # Probit estimation.            
        sys.stdout = open('/dev/null', 'w')
            
        rslt = sm.Probit(D, Z)
        P    = rslt.predict(rslt.fit().params)
            
        sys.stdout = sys.__stdout__
            
        # Determine common support.
        lowerBound = np.round(max(min(P[D == 1]), min(P[D == 0])), decimals = 2)
        upperBound = np.round(min(max(P[D == 1]), max(P[D == 0])), decimals = 2)
            
        # Finishing.
        return (lowerBound, upperBound)
    
    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''
        
        # numAgents.
        assert (isinstance(self.attr['numAgents'], int))
        assert (self.attr['numAgents'] > 0)
        
        # paraObjs.
        assert (all(isinstance(paraObj, _paraContainer) \
                for paraObj in self.attr['paraObjs']))

        # numSims.
        assert (isinstance(self.attr['numSims'], int))
        assert (self.attr['numSims'] > 0)
                
        # numParas.
        assert (isinstance(self.attr['numParas'], int))
        assert (self.attr['numParas'] > 0)
        
        # numFree.
        assert (isinstance(self.attr['numFree'], int))
        assert (self.attr['numFree'] > 0)

        # xExPost.
        assert (isinstance(self.attr['xExPost'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExPost'])))
        assert (self.attr['xExPost'].dtype == 'float')
        assert (self.attr['xExPost'].ndim == 2)      
 
        # xExAnte.
        assert (isinstance(self.attr['xExAnte'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExAnte'])))
        assert (self.attr['xExAnte'].dtype == 'float')
        assert (self.attr['xExAnte'].ndim == 2)    
        
        # factor.
        if(self.attr['factor'] is not None):
            
            assert (isinstance(self.attr['factor'], np.ndarray))
            assert (np.all(np.isfinite(self.attr['factor'])))
            assert (self.attr['factor'].dtype == 'float')
            assert (self.attr['factor'].ndim == 2)     
        
        # numCovarsExclBeneExAnte.
        assert (isinstance(self.attr['numCovarsExclBeneExAnte'], int))
        assert (self.attr['numCovarsExclBeneExAnte'] >= 0)

        # numCovarsExclBeneExPost.
        assert (isinstance(self.attr['numCovarsExclBeneExPost'], int))
        assert (self.attr['numCovarsExclBeneExPost'] >= 0)

        # numCovarsBeneExPost.
        assert (isinstance(self.attr['numCovarsBeneExPost'], int))
        assert (self.attr['numCovarsBeneExPost'] > 0)

        # numCovarsBeneExAnte.
        assert (isinstance(self.attr['numCovarsBeneExAnte'], int))
        assert (self.attr['numCovarsBeneExAnte'] > 0)

        # numCovarsExclCost.
        assert (isinstance(self.attr['numCovarsExclCost'], int))
        assert (self.attr['numCovarsExclCost'] > 0)

        # numCovarsOutc.
        assert (isinstance(self.attr['numCovarsOutc'], int))
        assert (self.attr['numCovarsOutc'] > 0)
        
        # numCovarsChoc.
        assert (isinstance(self.attr['numCovarsChoc'], int))
        assert (self.attr['numCovarsChoc'] > 0)
        
        # numCovarsCost.
        assert (isinstance(self.attr['numCovarsCost'], int))
        assert (self.attr['numCovarsCost'] > 0)    
    
        # numCoeffsBene.
        assert (isinstance(self.attr['numCoeffsBene'], int))
        assert (self.attr['numCoeffsBene'] > 0)    
        
        # Normalization.
        if(self.attr['normalization'] is not None):
            
            assert (isinstance(self.attr['normalization'], float))
            assert (self.attr['normalization'] > 0.0)
        
        # isNormalized.
        assert (self.attr['isNormalized'] in [True, False])

        # withoutPrediction.
        assert (self.attr['withoutPrediction'] in [True, False])

        # xExPostEval.
        assert (isinstance(self.attr['xExPostEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExPostEval'])))
        assert (self.attr['xExPostEval'].dtype == 'float')
        assert (self.attr['xExPostEval'].ndim == 1)    

        # xExAnteEval.
        assert (isinstance(self.attr['xExAnteEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['xExAnteEval'])))
        assert (self.attr['xExAnteEval'].dtype == 'float')
        assert (self.attr['xExAnteEval'].ndim == 1)            

        # zEval
        assert (isinstance(self.attr['zEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['zEval'])))
        assert (self.attr['zEval'].dtype == 'float')
        assert (self.attr['zEval'].ndim == 1)    
        
        # cEval.
        assert (isinstance(self.attr['cEval'], np.ndarray))
        assert (np.all(np.isfinite(self.attr['cEval'])))
        assert (self.attr['cEval'].dtype == 'float')
        assert (self.attr['cEval'].ndim == 1)            
        
        # Finishing.
        return True

''' Private methods and classes of the module. 
'''
class _paraContainer(object):
    ''' Container for parameter class.
    '''
    def __init__(self):
        ''' Parameter initialization.
        '''
        
        # Attach attributes.
        self.attr = {}
        
        self.attr['hasBounds'] = False      
        
        self.attr['subgroup']  = None          
        self.attr['type']      = None
        self.attr['value']     = None        
        self.attr['isFree']    = None        
        self.attr['pvalue']    = None          
        self.attr['startVal']  = None        
        
        self.attr['confi']     = (None, None)        
        self.attr['bounds']    = (None, None)        

        self.isLocked          = False
        
    ''' Public get/set methods.
    '''
    def getAttr(self, key):
        ''' Get attribute.
        '''
        # Antibugging. 
        assert (self._checkKey(key) == True)
        assert (self.getStatus() == True)
        
        # Select from dictionary.
        return self.attr[key]   

    def setAttr(self, key, arg):
        ''' Set attribute.
        '''
        # Antibugging.
        assert (self._checkKey(key) == True)
        
        # Set attribute.
        self.attr[key] = arg
    
    def lock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == False)
        
        # Process attributes.
        self._updateAttributes()

        # Update status.
        self.isLocked = True
                
        # Check integrity.
        assert (self._checkIntegrity() == True)
    
    def getStatus(self):
        ''' Get status of class instance.
        '''
        # Return. 
        return self.isLocked       
        
    ''' Private methods
    '''
    def _updateAttributes(self):
        ''' Update endogenous attributes.
        '''
        
        if(np.any(self.attr['bounds']) is not None):
            
            self.attr['hasBounds'] = True
    
    def _checkIntegrity(self):
        ''' Check integrity.
        '''
        # type.
        assert (self.getAttr('type') in ['outc', 'cost', 'rho', 'sd'])

        # subgroup.
        if(self.getAttr('subgroup') is not None):
            
            assert (isinstance(self.getAttr('subgroup'), str))
        
        # value.
        assert (isinstance(self.getAttr('value'), float))
        assert (np.isfinite(self.getAttr('value')))           

        # startVal.
        assert (isinstance(self.getAttr('startVal'), float))
        assert (np.isfinite(self.getAttr('startVal')))      
                
        # isFree.
        assert (self.getAttr('isFree') in [True, False])
        
        # hasBounds.
        assert (isinstance(self.attr['bounds'], tuple))
        assert (len(self.attr['bounds']) == 2)
        
        # value.
        assert (isinstance(self.attr['value'], float))
        assert (np.isfinite(self.attr['value']))

        # confi.
        assert (isinstance(self.attr['confi'], tuple))
        assert (len(self.attr['confi']) == 2)

        # pvalue.
        if(self.attr['pvalue'] is not None):
            
            assert (isinstance(self.attr['pvalue'], float))
            assert (np.isfinite(self.attr['pvalue']))
            
        # isLocked
        assert (self.isLocked in [True, False])
        
        # Finishing.
        return True
    
    def _checkKey(self, key):
        ''' Check that key is present.
        '''        
        # Check presence.
        assert (key in self.attr.keys())
        
        # Finishing.
        return True
        