''' Module that holds the parasCls, which manages all things related to the
    parameter management.
'''
# standard library
import numpy   as np

# project library
from grmpy.clsMeta import MetaCls
from grmpy.clsModel import ModelCls

class parasCls(MetaCls):
    ''' Class for the parameter management.
    '''
    def __init__(self, modelObj):
        
        # Antibugging.
        assert (isinstance(modelObj, ModelCls))
        assert (modelObj.get_status() is True)
        
        # Attach attributes.
        self.attr = {}

         
        self.attr['paraObjs']   = []
        self.attr['numParas']   = 0
        self.attr['numFree']    = 0
        self.attr['factor']     = None

        self.attr['num_covars_excl_bene_ex_ante'] = modelObj.get_attr('num_covars_excl_bene_ex_ante')
        self.attr['num_covars_excl_cost']       = modelObj.get_attr('num_covars_excl_cost')

        self.attr['without_prediction']       = modelObj.get_attr('without_prediction')
        self.attr['surp_estimation']          = modelObj.get_attr('surp_estimation')

        self.attr['X_ex_ante']                 = modelObj.get_attr('X_ex_ante')
        self.attr['X_ex_post']                 = modelObj.get_attr('X_ex_post')

        self.attr['num_agents']               = modelObj.get_attr('num_agents')
        
        # Initialization.
        self.attr['modelObj']  = modelObj    
        
        self.isFirst = True
    
        # Status.
        self.is_locked = False
    
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
            
            paraObj.set_attr('id', id_)
            
        else:
            
            paraObj.set_attr('id', None)

        paraObj.set_attr('col', col)
        
        paraObj.set_attr('count', count)
                        
        paraObj.set_attr('type', type_)
        
        paraObj.set_attr('subgroup', subgroup)
        
        paraObj.set_attr('value', value)
        
        paraObj.set_attr('isFree', isFree)

        paraObj.set_attr('bounds', bounds)
                     
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
        assert (self.get_status() == True)
        assert (count < self.get_attr('numParas'))
        
        # Algorithm.
        paraObjs = self.get_attr('paraObjs')

        for paraObj in paraObjs:
            
            paraCount = paraObj.get_attr('count')
        
            if(paraCount == count):
                
                return paraObj
    
        # Finishing.
        assert(False == True)

    def getParameters(self, type_, subgroup, isObj = False):
        ''' Get parameter groups.
        '''
        # Antibugging.
        assert (self._checkRequest(type_, subgroup, isObj) == True)
            
        # Collect request.
        rsltList = []
            
        if(type_ in ['outc', 'cost', 'rho', 'sd']):
            
            for paraObj in self.attr['paraObjs']:
                
                if(paraObj.get_attr('type')      != type_):    continue
                
                if(paraObj.get_attr('subgroup')  != subgroup): continue
                
                if(isObj):
                    
                    rsltList.append(paraObj)
        
                else:
                    
                    rsltList.append(paraObj.get_attr('value'))
        
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
            
            numCovarsExclCost       = self.get_attr('num_covars_excl_cost')
            numCovarExclBeneExAnte  = self.get_attr(
                'num_covars_excl_bene_ex_ante')
            
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
        assert (self.get_status() == True)
        assert (self._checkIntegrity() == True)
        assert (version in ['external', 'internal'])
        assert (which in ['free', 'all'])

        # Main algorithm.
        paraObjs = self.attr['paraObjs']
        
        rslt = []
        
        for paraObj in paraObjs:
            
            isFixed = (paraObj.get_attr('isFree') == False)
            
            if(isFixed and (which == 'free')): continue
                        
            value = paraObj.get_attr('value')
            
            if(version == 'external'):

                value = self._transformToExternal(paraObj, paraObj.get_attr('value'))
                
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
        assert (self.get_status() == True)
        assert (self._checkIntegrity() == True)
        assert (version in ['external', 'internal'])
        assert (which in ['free', 'all'])
        
        assert (isinstance(x, np.ndarray))
        assert (np.all(np.isfinite(x)))
        assert (x.dtype == 'float')
        
        if(which == 'all'):
            
            assert (x.shape == (self.get_attr('numParas'), ))
        
        else:
            
            assert (x.shape == (self.get_attr('numFree'), ))
        
        # Distribute class attributes.
        paraObjs = self.get_attr('paraObjs')
        
        counter = 0
        
        for paraObj in paraObjs:
           
            isFixed = (paraObj.get_attr('isFree') == False)
            
            if(isFixed and (which == 'free')): continue
            
            value = x[counter]
 
            if(paraObj.get_attr('hasBounds') and (version == 'external')):
                
                value = self._transformToInternal(paraObj, value)
            
            paraObj.set_value(value)
            
            counter += 1
        
        # Finishing.
        return True
    
    def _transformToExternal(self, paraObj, internalValue):
        ''' Transform internal values for external use by maximization 
            routine.
        '''
        # Antibugging.
        assert (isinstance(paraObj, _paraContainer))
        assert (paraObj.get_status() == True)
        assert (isinstance(internalValue, float))
        assert (np.isfinite(internalValue))
        
        # Auxiliary objects.
        lowerBound, upperBound = paraObj.get_attr('bounds')
        
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
        assert (paraObj.get_status() == True)
        assert (isinstance(externalValue, float))
        assert (np.isfinite(externalValue))
        
        # Auxiliary objects.
        lowerBound, upperBound = paraObj.get_attr('bounds')
        
        hasBounds = paraObj.get_attr('hasBounds')
        
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
        assert (paraObj.get_status() == True)
        assert (isinstance(internalValue, float))
        assert (np.isfinite(internalValue))
                
        # Auxiliary objects.
        lowerBound, upperBound = paraObj.get_attr('bounds')

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
        assert (self.get_status() == True)

        # Distribute class attributes.
        without_prediction = self.get_attr('without_prediction')
        coeffsBeneExPost  = self.getParameters('bene', 'exPost')
        
        # Check applicability.
        if(without_prediction): return coeffsBeneExPost 
        
        x_ex_post = self.get_attr('X_ex_post')
        x_ex_ante = self.get_attr('X_ex_ante')
    
        # Construct index.       
        idxBene = np.dot(coeffsBeneExPost, x_ex_post.T)
        
        if(self.attr['factor'] is None):
     
            pinv = np.linalg.pinv(np.dot(x_ex_ante.T, x_ex_ante))
            
            self.attr['factor'] = np.dot(pinv, x_ex_ante.T)
                
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
            
            paraObj.set_attr('id', None)
             
            paraObj.lock()

            if(paraObj.get_attr('isFree')):
                
                paraObj.set_attr('id', id_)
                
                id_ += 1

        self.unlock()
        
        self.set_attr('numFree', id_)
        
        self.lock()

    ''' Check integrity of class instance and attribute requests.
    '''
    def _checkRequest(self, type_, subgroup, obj):
        ''' Check the validity of the parameter request.
        '''
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
class _paraContainer(MetaCls):
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

        self.is_locked          = False
        
    ''' Public get/set methods.
    '''
    def set_value(self, arg):
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
        
    def set_attr(self, key, arg):
        ''' Set attribute.
        
            Development Note:
            
                This function overrides the metaCls method. Otherwise, 
                the updating step during estimation is too tedious.
        
        '''
        # Antibugging.
        assert (self.check_key(key) == True)

        # Set attribute.
        self.attr[key] = arg

    ''' Private methods
    '''
    def derived_attributes(self):
        ''' Update endogenous attributes.
        '''
        
        if(np.any(self.attr['bounds']) is not None):
            
            self.attr['hasBounds'] = True
    
    def _check_integrity(self):
        ''' Check integrity.
        '''
        # type.
        assert (self.get_attr('type') in ['outc', 'cost', 'rho', 'sd'])

        # column, 
        if(self.get_attr('col') is not None):

            col = self.get_attr('col')

            assert (isinstance(col, int) or (col == 'int'))
            if col != 'int': assert (col >= 0)

        # subgroup.
        if(self.get_attr('subgroup') is not None):
            
            assert (isinstance(self.get_attr('subgroup'), str))
        
        # value.
        assert (isinstance(self.get_attr('value'), float))
        assert (np.isfinite(self.get_attr('value')))

        # isFree.
        assert (self.get_attr('isFree') in [True, False])
        
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
        assert (self.is_locked in [True, False])
        
        # Finishing.
        return True

        