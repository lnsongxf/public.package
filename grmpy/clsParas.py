''' Module that holds the parasCls, which manages all things related to the
    parameter management.
'''
# standard library
import numpy   as np

# project library
from grmpy.clsMeta import metaCls
import grmpy.clsModel

class parasCls(metaCls):
    ''' Class for the parameter management.
    '''
    def __init__(self, modelObj):
        
        # Antibugging.
        assert (isinstance(modelObj, grmpy.clsModel.modelCls))
        assert (modelObj.getStatus() == True)
        
        # Attach attributes.
        self.attr = {}

         
        self.attr['paraObjs']   = []
        self.attr['numParas']   = 0
        self.attr['numFree']    = 0
        self.attr['factor']     = None

        self.attr['numCovarsExclBeneExAnte'] = modelObj.getAttr('numCovarsExclBeneExAnte')                    
        self.attr['numCovarsExclCost']       = modelObj.getAttr('numCovarsExclCost')

        self.attr['withoutPrediction']       = modelObj.getAttr('withoutPrediction')
        self.attr['surpEstimation']          = modelObj.getAttr('surpEstimation')

        self.attr['xExAnte']                 = modelObj.getAttr('xExAnte')
        self.attr['xExPost']                 = modelObj.getAttr('xExPost')

        self.attr['numAgents']               = modelObj.getAttr('numAgents')
        
        # Initialization.
        self.attr['modelObj']  = modelObj    
        
        self.isFirst = True
    
        # Status.
        self.isLocked = False
    
    def addParameter(self, type_, subgroup, value, isFree, bounds, col):
        ''' Add parameters to class instance.
        '''
        # Antibugging.
        assert (isFree in  [True, False])
        assert (len(bounds) == 2)
        assert (type_ in ['outc', 'cost', 'sd', 'rho'])
        
        if(type_ in ['outc', 'cost']):
            
            assert (isinstance(col, int) or (col == 'int'))

        if(type_ == 'sd'):
            
            assert (bounds[0] is not None)
            assert (bounds[0] > 0.0)
            assert (col is None)
            
        if(type_ == 'rho'):
            
            assert (bounds[0] > -1.00)
            assert (bounds[1] <  1.00)
            assert (col is None)
            
        # Initialize parameters.
        count   = self.attr['numParas']
        
        paraObj = _paraContainer()

        if(isFree):

            id_   = self.attr['numFree']
            
            paraObj.setAttr('id', id_)
            
        else:
            
            paraObj.setAttr('id', None)

        paraObj.setAttr('col', col)
        
        paraObj.setAttr('count', count)
                        
        paraObj.setAttr('type', type_)
        
        paraObj.setAttr('subgroup', subgroup)
        
        paraObj.setAttr('value', value)
        
        paraObj.setAttr('isFree', isFree)

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
        paraObjs = self.getAttr('paraObjs')

        for paraObj in paraObjs:
            
            paraCount = paraObj.getAttr('count')
        
            if(paraCount == count):
                
                return paraObj
    
        # Finishing.
        assert(False == True)

    def getParameters(self, type_, subgroup, isObj = False):
        ''' Get parameter groups.
        '''
        # Antibugging.
        #assert (self.getStatus() == True)
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
            
            numCovarsExclCost       = self.getAttr('numCovarsExclCost')
            numCovarExclBeneExAnte  = self.getAttr('numCovarsExclBeneExAnte')
            
            coeffsBeneExAnte = self.getParameters('bene', 'exAnte')   
            coeffsCost       = self.getParameters('cost', None)   

            coeffsBene = np.concatenate((coeffsBeneExAnte, np.tile(0.0, numCovarsExclCost)))
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

    def getValues(self, version, which):
        ''' Get all free parameter values.
        '''    
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkIntegrity() == True)
        assert (version in ['external', 'internal'])
        assert (which in ['free', 'all'])

        # Main algorithm.
        paraObjs = self.attr['paraObjs']
        
        rslt = []
        
        for paraObj in paraObjs:
            
            isFixed = (paraObj.getAttr('isFree') == False)
            
            if(isFixed and (which == 'free')): continue
                        
            value = paraObj.getAttr('value')
            
            if(version == 'external'):

                value = self._transformToExternal(paraObj, paraObj.getAttr('value'))
                
            rslt.append(value)
            
        # Type conversion.
        rslt = np.array(rslt)
        
        # Quality checks.
        assert (isinstance(rslt, np.ndarray))
        assert (np.all(np.isfinite(rslt)))
        assert (rslt.dtype == 'float')
        
        if(which == 'all'): 
            
            assert (rslt.shape == (self.attr['numParas'], ))
        
        else:
            
            assert (rslt.shape == (self.attr['numFree'], ))
            
        # Finishing.
        return rslt

    ''' All methods related to updating the parameters. 
    '''             
    def update(self, x, version, which):
        ''' Update all free parameters.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (self._checkIntegrity() == True)
        assert (version in ['external', 'internal'])
        assert (which in ['free', 'all'])
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        
        if(which == 'all'):
            
            assert (x.shape == (self.getAttr('numParas'), ))
        
        else:
            
            assert (x.shape == (self.getAttr('numFree'), ))
        
        # Distribute class attributes.
        paraObjs = self.getAttr('paraObjs')
        
        counter = 0
        
        for paraObj in paraObjs:
           
            isFixed = (paraObj.getAttr('isFree') == False)
            
            if(isFixed and (which == 'free')): continue
            
            value = x[counter]
 
            if(paraObj.getAttr('hasBounds') and (version == 'external')):
                
                value = self._transformToInternal(paraObj, value)
            
            paraObj.setValue(value)
            
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
        
        # Finishing.
        return rslt
    
    def _replaceParasObj(self, paraObj):
        ''' Replace parameter object.
        '''
        # Update attributes.
        parasList = self.attr['paraObjs']
        
        id_ = 0
        
        for paraObj in parasList:
            
            paraObj.unlock()
            
            paraObj.setAttr('id', None)
             
            paraObj.lock()

            if(paraObj.getAttr('isFree')):
                
                paraObj.setAttr('id', id_)
                
                id_ += 1

        self.unlock()
        
        self.setAttr('numFree', id_)
        
        self.lock()

    ''' Check integrity of class instance and attribute requests.
    '''
    def _checkRequest(self, type_, subgroup, obj):
        ''' Check the validity of the parameter request.
        '''
        # Antibugging.
        #assert (self.getStatus() == True)
     
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
    
    def _checkIntegrity(self):
        
        return True

''' Private methods and classes of the module. 
'''
class _paraContainer(metaCls):
    ''' Container for parameter class.
    '''
    counter = 0
    
    def __init__(self):
        ''' Parameter initialization.
        '''
        
        # Attach attributes.
        self.attr = {}

        self.attr['id']     = None
        self.attr['col']    = None

        self.attr['count']  = None
        
        self.attr['bounds']    = (None, None)        
          
        self.attr['subgroup']  = None          
        self.attr['type']      = None
        self.attr['value']     = None        
        self.attr['isFree']    = None        
        
        self.attr['pvalue']    = None          
        self.attr['confi']     = (None, None)        

        self.attr['hasBounds'] = False      

        self.isLocked          = False
        
    ''' Public get/set methods.
    '''
    def setValue(self, arg):
        ''' Set value of parameter object.
        '''
        # Antibugging.
        assert (isinstance(arg, float))
        
        # Distribute class attributes.
        isFree       = self.attr['isFree']
    
        value        = self.attr['value']
        
        hasBounds    = self.attr['hasBounds']
            
        lower, upper = self.attr['bounds']
            
        # Checks.        
        if(not isFree): assert (value == arg)
            
        if(hasBounds):
                
            if(lower is not None): assert (value > lower)
        
            if(upper is not None): assert (value < upper)
        
        # Set attribute.
        self.attr['value'] = arg
        
    def setAttr(self, key, arg):
        ''' Set attribute.
        
            Development Note:
            
                This function overrides the metaCls method. Otherwise, 
                the updating step during estimation is too tedious.
        
        '''
        # Antibugging.
        assert (self._checkKey(key) == True)

        # Set attribute.
        self.attr[key] = arg

    ''' Private methods
    '''
    def _derivedAttributes(self):
        ''' Update endogenous attributes.
        '''
        
        if(np.any(self.attr['bounds']) is not None):
            
            self.attr['hasBounds'] = True
    
    def _checkIntegrity(self):
        ''' Check integrity.
        '''
        # type.
        assert (self.getAttr('type') in ['outc', 'cost', 'rho', 'sd'])

        # column, 
        if(self.getAttr('col') is not None):

            col = self.getAttr('col')
            
            assert (isinstance(col, int) or (col == 'int'))
            assert (self.getAttr('col') >= 0)

        # subgroup.
        if(self.getAttr('subgroup') is not None):
            
            assert (isinstance(self.getAttr('subgroup'), str))
        
        # value.
        assert (isinstance(self.getAttr('value'), float))
        assert (np.isfinite(self.getAttr('value')))           

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

        