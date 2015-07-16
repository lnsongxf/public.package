''' Module contains all functions required to generate a parameter object
    from the processed initialization file.
'''

# standard library
import statsmodels.api  as sm
import numpy            as np

import sys

# project library.
from grmpy.clsParas import ParasCls
from grmpy.clsModel import ModelCls


''' Main function.
'''
def constructParas(initDict, modelObj, isSimulation):
    ''' Construct parameter object.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (modelObj.get_status() == True)

    # Distribute auxiliary objects.
    start = initDict['ESTIMATION']['start']
    
    # Initialize with manual starting values.
    parasObj = _initializeParameters(initDict, modelObj)
        
    # Update with automatic starting values.
    if(start == 'auto' and (not isSimulation)):
    
        parasObj = _autoStart(parasObj, modelObj)
                
    # Quality.
    assert (parasObj.get_status() == True)

    # Finishing.
    return parasObj

''' Private auxiliary functions.
'''
def _initializeParameters(initDict, modelObj):
    ''' Get starting values from initialization file.
    '''
    def _getValues(group, subgroup, initDict):
        ''' Order the starting values such that they are matched with the correct
            columns. This includes the intercept.
        '''
        def _collectInformation(positions, dict_):
            ''' Order the information appropriately.
            '''
            # Antibugging.
            assert (isinstance(positions, list))
            assert (isinstance(dict_, dict))
            assert (set(dict_.keys()) ==  set(positions))
            
            # Initialize containers.
            values  = []
            
            isFrees = []
            
            cols    = []
            
            # Collect info.
            for pos in positions:
                
                values  += [dict_[pos]['value']]
         
                isFrees += [dict_[pos]['isFree']]
                
                cols    += [dict_[pos]['col']]
        
            # Quality.
            assert (all(isinstance(value, float) for value  in values))
            assert (all(isinstance(isFree, bool) for isFree in isFrees))
    
            # Finishing.
            return values, cols, isFrees
            
        # Antibugging.
        assert (isinstance(initDict, dict))
        assert (group in ['BENE', 'COST'])
        
        if(group ==  'BENE'): assert (subgroup in ['TREATED', 'UNTREATED'])
    
        if(group ==  'COST'): assert (subgroup is None)
    
        # Distribute information.
        common         = initDict['DERIV']['common']['pos']
        
        exclBeneExPost = initDict['DERIV']['excl_bene']['ex_post']['pos']
    
        exclCost       = initDict['DERIV']['excl_cost']['pos']
        
        # Initialize container.
        dict_ = {}
        
        # Benefits.
        if(group == 'BENE'):
            
            # Coefficients.
            isFrees   = initDict['BENE'][subgroup]['coeffs']['free'][:]
    
            values    = initDict['BENE'][subgroup]['coeffs']['values'][:]
            
            positions = initDict['BENE'][subgroup]['coeffs']['pos'][:]
            
            for pos  in positions:
                
                dict_[pos] = {}
                
                dict_[pos]['value']  = values.pop(0)
    
                dict_[pos]['isFree'] = isFrees.pop(0)
                
                dict_[pos]['col']    = pos
                   
            # Intercept.
            dict_['int'] = {}
            
            dict_['int']['value']  = initDict['BENE'][subgroup]['int']['values'][0]
    
            dict_['int']['isFree'] = initDict['BENE'][subgroup]['int']['free'][0]
    
            dict_['int']['col']    = 'int'
    
            # Collect in order.
            positions = exclBeneExPost + common + ['int']
           
        # Costs.
        if(group == 'COST'):
            
            # Coefficients.
            isFrees   = initDict['COST']['coeffs']['free'][:]
    
            values    = initDict['COST']['coeffs']['values'][:]
            
            positions = initDict['COST']['coeffs']['pos'][:]
            
            for pos  in positions:
                
                dict_[pos] = {}
                
                dict_[pos]['value']  = values.pop(0)
    
                dict_[pos]['isFree'] = isFrees.pop(0)
                
                dict_[pos]['col']    = pos 
            
            # Intercept.
            dict_['int'] = {}
            
            dict_['int']['value']  = initDict['COST']['int']['values'][0]
    
            dict_['int']['isFree'] = initDict['COST']['int']['free'][0]
    
            dict_['int']['col']    = 'int'
    
            
            positions = common + ['int'] + exclCost
    
        # Create output.
        values, cols, isFrees = _collectInformation(positions, dict_)
        
        # Finishing.
        return values, cols, isFrees
    
    ''' Core function.
    ''' 
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (modelObj.get_status() == True)

    # Distribute information
    numCovarsExPost = len(initDict['BENE']['TREATED']['coeffs']['pos'])

    numCovarsCost   = len(initDict['COST']['coeffs']['pos'])

    # Initialize parameter container.
    parasObj = ParasCls(modelObj)

    # Benefits.
    
        # Treated
    values, cols, isFrees  = _getValues('BENE', 'TREATED', initDict)

    for i in range(numCovarsExPost + 1):
    
        type_    = 'outc'
        
        subgroup = 'treated'
        
        value    = values[i]
        
        col      = cols[i]
                
        isFree   = isFrees[i]
        
        parasObj.add_parameter(type_, subgroup, value, is_free= isFree, bounds = (None, None), col = col)
    
        # Untreated
    values, cols, isFrees  = _getValues('BENE', 'UNTREATED', initDict)

    for i in range(numCovarsExPost + 1):
    
        type_    = 'outc'
        
        subgroup = 'untreated'
        
        value    = values[i]
        
        col      = cols[i]
        
        isFree   = isFrees[i]
        
        parasObj.add_parameter(type_, subgroup, value, is_free= isFree, bounds = (None, None), col = col)
    
    # Costs.
    values, cols, isFrees  = _getValues('COST', None, initDict)
    
    for i in range(numCovarsCost + 1):
    
        type_    = 'cost'
                
        value    = values[i]
        
        col      = cols[i]
        
        isFree   = isFrees[i]
        
        parasObj.add_parameter(type_, None, value, is_free= isFree, bounds = (None, None), col = col)

    # Correlation parameters.
    value  = initDict['RHO']['treated']['value']
    isFree = initDict['RHO']['treated']['free']
    
    parasObj.add_parameter('rho', 'U1,V', value, isFree, (-0.99, 0.99), col = None)
    
    value  = initDict['RHO']['untreated']['value']
    isFree = initDict['RHO']['untreated']['free']

    parasObj.add_parameter('rho', 'U0,V', value, isFree, (-0.99, 0.99), col = None)
    
    # Disturbances.
    value  = initDict['BENE']['UNTREATED']['sd']['values'][0]
    isFree = initDict['BENE']['UNTREATED']['sd']['free'][0]
            
    parasObj.add_parameter('sd', 'U0', value, is_free= isFree, bounds = (0.01, None), col = None)
   
   
    value  = initDict['BENE']['TREATED']['sd']['values'][0]
    isFree = initDict['BENE']['TREATED']['sd']['free'][0]
      
    parasObj.add_parameter('sd', 'U1', value, is_free= isFree, bounds = (0.01, None), col = None)

    
    value  = initDict['COST']['sd']['values'][0]
    isFree = initDict['COST']['sd']['free'][0]
        
    parasObj.add_parameter('sd', 'V', value, is_free= isFree, bounds = (0.01, None), col = None)
        
    parasObj.lock()

    # Finishing.   
    return parasObj
    
def _autoStart(parasObj, modelObj):
    ''' Get automatic starting values.
    '''   
    def _computeStartingValues(modelObj, which):
        ''' Get starting values.
        '''
        # Antibugging.
        assert (modelObj.get_status() == True)
        assert (which in ['treated', 'untreated', 'cost'])
        
        # Data selection.
        Y = modelObj.get_attr('Y')

        D = modelObj.get_attr('D')
        
        
        X = modelObj.get_attr('X_ex_post')
        
        G = modelObj.get_attr('G')
        
        # Subset selection.
        if(which == 'treated'):
            
            Y = Y[D == 1]
            
            X = X[(D == 1), :]
        
        elif(which == 'untreated'):
            
            Y = Y[D == 0]
            
            X = X[(D == 0), :]
        
        # Model selection. 
        if(which in ['treated', 'untreated']):
            
            olsRslt = sm.OLS(Y, X).fit()

            coeffs = olsRslt.params
            sd     = np.array(np.sqrt(olsRslt.scale))
        
        elif(which == 'cost'):
            
            sys.stdout = open('/dev/null', 'w')
            
            probitRslt =  sm.Probit(D, G).fit()
            
            coeffs = -probitRslt.params
            sd     = np.array(1.0)
            
            sys.stdout = sys.__stdout__
            
        # Quality checks.
        assert (isinstance(coeffs, np.ndarray))
        assert (isinstance(sd, np.ndarray))
        
        assert (np.all(np.isfinite(coeffs)))                
        assert (np.all(np.isfinite(sd)))
    
        assert (coeffs.ndim == 1)
        assert (sd.ndim == 0)
        
        assert (coeffs.dtype == 'float')
        assert (sd.dtype == 'float')

        # Type conversions.
        coeffs = coeffs.tolist()
        sd     = float(sd)
        
        # Finishing.
        return coeffs, sd
    
    ''' Core function.
    '''
    # Antibugging.
    assert (isinstance(parasObj, ParasCls))
    assert (parasObj.get_status() == True)

    assert (isinstance(modelObj, ModelCls))
    assert (modelObj.get_status() == True)
    
    # Benefits.
    for subgroup in ['treated', 'untreated']:

        paraObjs = parasObj.get_parameters('outc', subgroup, is_obj = True)
        
        coeffs, sd = _computeStartingValues(modelObj, subgroup)
        
        assert (len(paraObjs) == len(coeffs))
        
        for paraObj in paraObjs:
            
            coeff = coeffs.pop(0)
            
            paraObj.set_attr('value', coeff)
        
            parasObj._replace_paras_obj(paraObj)
        
        label = 'U1'
        
        if(subgroup == 'untreated'): label = 'U0'
        
        paraObj = parasObj.get_parameters('sd', label, is_obj = True)
        
        paraObj.set_attr('value', sd)
        
        parasObj._replace_paras_obj(paraObj)

    # Cost.
    paraObjs = parasObj.get_parameters('cost', None, is_obj = True)
        
    coeffs, sd = _computeStartingValues(modelObj, 'cost')
        
    assert (len(paraObjs) == len(coeffs))
            
    for paraObj in paraObjs:
            
        coeff = coeffs.pop(0)
            
        paraObj.set_attr('value', coeff)
        
        parasObj._replace_paras_obj(paraObj)
    
    
    paraObj = parasObj.get_parameters('sd', 'V', is_obj = True)
        
    paraObj.set_attr('value', sd)
        
    parasObj._replace_paras_obj(paraObj)
    
    # Correlations.
    for corr in ['U1,V', 'U0,V']:

        paraObj = parasObj.get_parameters('rho', corr, is_obj = True)
            
        paraObj.set_attr('value', 0.0)
            
        parasObj._replace_paras_obj(paraObj)
    

    # Quality.
    assert (parasObj.get_status() == True)
    
    # Finishing.
    return parasObj
    