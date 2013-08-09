''' This module contains the interface to the maximization routine for the
    grmEstimatorToolbox.
'''
# standard library
import  sys

import  numpy           as      np
import  numdifftools    as      nd

from    scipy.optimize  import  fmin_bfgs

# project library
import clsUser
import clsRslt

import modMaxFunctions

def maximize(userRequest):
    ''' Perform the requested maximization given the user's
        specification.
    '''
    # Antibugging.
    assert (isinstance(userRequest, clsUser.userRequest))
    assert (userRequest.getStatus() == True)
    
    # Set random seed.
    np.random.seed(123)
    
    # Distribute class attributes.
    parasObj            = userRequest.getAttr('parasObj')   
    
    maxiter             = userRequest.getAttr('maxiter')
    
    numAgents           = userRequest.getAttr('numAgents')
    
    alpha               = userRequest.getAttr('alpha') 
        
    surpEstimation      = userRequest.getAttr('surpEstimation')

    hessian             = userRequest.getAttr('hessian')

    numDraws            = userRequest.getAttr('numDraws')

    isDebug             = userRequest.getAttr('isDebug') 

    epsilon             = userRequest.getAttr('epsilon') 

    gtol                = userRequest.getAttr('gtol') 

    withAverageEffects  = userRequest.getAttr('withAverageEffects') 

    withMarginalEffects = userRequest.getAttr('withMarginalEffects') 

    withConditionalEffects = userRequest.getAttr('withConditionalEffects') 
        
    # Distribute evaluation points.
    xExPostEval = userRequest.getAttr('xExPostEval') 

    xExAnteEval = userRequest.getAttr('xExAnteEval') 
    
    zEval       = userRequest.getAttr('zEval')

    cEval       = userRequest.getAttr('cEval')

    P           = userRequest.getAttr('P')
    
    D           = userRequest.getAttr('D')
                                        
    # Distribute auxiliary objects.
    startingValues = parasObj.getValues()

    # Algorithm.
    sys.stdout = open('logFile.log', 'w')   
        
    maxRslt = fmin_bfgs(_scipyWrapper, startingValues, args = (userRequest,), 
                    full_output = True, maxiter = maxiter, epsilon = epsilon,
                    gtol = gtol)

    sys.stdout = sys.__stdout__

    # Distribute results.
    xopt      = maxRslt[0]
    fun       = maxRslt[1]
    grad      = maxRslt[2]
    message   = maxRslt[6]
    
    isSuccess = (maxRslt[6] == 0)
    
    # Check success.
    if(isDebug == False):
        
        assert (isSuccess == True)
    
    # Approximate hessian.
    if(hessian == 'bfgs'):
        
        covMat = maxRslt[3]
        
    else:
        
        ndObj   = nd.Hessian(lambda x: _scipyWrapper(x, userRequest)) 
        hess    = ndObj(xopt)
        covMat  = np.linalg.pinv(hess)

    # Distribute arguments.
    parasObj = userRequest.getAttr('parasObj')   
        
    # Construct result class.
    rslt = clsRslt.results()
    
    rslt.setAttr('withAverageEffects', withAverageEffects)

    rslt.setAttr('withConditionalEffects', withConditionalEffects)

    rslt.setAttr('withMarginalEffects', withMarginalEffects)

    rslt.setAttr('hessian', hessian)

    rslt.setAttr('message', message)
            
    rslt.setAttr('xopt', xopt)

    rslt.setAttr('grad', grad)
    
    rslt.setAttr('fun', fun)
            
    rslt.setAttr('isSuccess', isSuccess)

    rslt.setAttr('maxiter', maxiter)

    rslt.setAttr('isDebug', isDebug)
    
    rslt.setAttr('numDraws', numDraws)
            
    rslt.setAttr('covMat', covMat)
            
    rslt.setAttr('parasObj', parasObj)
    
    rslt.setAttr('xExPostEval', xExPostEval)
        
    rslt.setAttr('xExAnteEval', xExAnteEval)
        
    rslt.setAttr('zEval', zEval)        
    
    rslt.setAttr('cEval', cEval)        

    rslt.setAttr('D', D)   
    
    rslt.setAttr('P', P)       
            
    rslt.setAttr('numAgents', numAgents)
    
    rslt.setAttr('alpha', alpha)
    
    rslt.setAttr('surpEstimation', surpEstimation)

    rslt.lock()

    rslt.store()
    
    # Finishing.
    return rslt

''' Private functions of the module.
'''
def _scipyWrapper(x, userRequest):
    ''' Wrapper for most scipy maximization algorithms.
    '''
    # Antibugging.
    assert (isinstance(userRequest, clsUser.userRequest))
    assert (userRequest.getStatus() == True)

    assert (isinstance(x, np.ndarray))
    assert (np.all(np.isfinite(x)))
    assert (x.dtype == 'float')
    assert (x.ndim == 1)

    # Distribute class attributes.    
    parasObj = userRequest.getAttr('parasObj')
    
    # Update parameter class.
    parasObj.updateValues(x)
    
    # Evaluate likelihood.
    likl = modMaxFunctions.criterionFunction(userRequest)
    
    # Quality checks.
    assert (isinstance(likl, float))    
    assert (np.isfinite(likl))
    assert (likl > 0.0)
    
    #Finishing.        
    return likl