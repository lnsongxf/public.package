''' This module contains the interface to the maximization routine for the
    grmEstimatorToolbox.
'''
# standard library
import  numpy           as      np
import  numdifftools    as      nd
import  cPickle         as      pkl

import  sys

# project library
import clsRequest
import clsRslt
import clsModel
import clsParas
import clsMax
import clsGrm

def maximize(modelObj, parasObj, requestObj):
    ''' Perform the requested maximization given the user's
        specification.
    '''
    # Antibugging.
    assert (isinstance(modelObj, clsModel.modelCls))
    assert (modelObj.getStatus() == True)
        
    assert (isinstance(parasObj, clsParas.parasCls))
    assert (parasObj.getStatus() == True)
        
    assert (isinstance(requestObj, clsRequest.requestCls))
    assert (requestObj.getStatus() == True)

    # Initialize container
    grmObj = clsGrm.grmCls()
    
    grmObj.setAttr('modelObj', modelObj)
    
    grmObj.setAttr('requestObj', requestObj)

    grmObj.setAttr('parasObj', parasObj)
    
    grmObj.lock()
    
    # Set random seed.
    np.random.seed(123)
    
    # Distribute class attributes.
    requestObj = grmObj.getAttr('requestObj')
    
    hessian   = requestObj.getAttr('hessian')

    withAsymptotics = requestObj.getAttr('withAsymptotics') 
                   
    # Distribute auxiliary objects.
    maxCls = clsMax.maxCls(grmObj)
    
    maxCls.lock()
     
    sys.stdout = open('maxReport.grm.log', 'w')   
    
    maxRslt = maxCls.maximize()
    
    sys.stdout = sys.__stdout__
        
    # Distribute results.
    xopt      = maxRslt['xopt']
    
    # Approximate hessian.
    covMat  = np.tile(np.nan, (len(xopt), len(xopt)))
    
    if(withAsymptotics):
        
        if(hessian == 'bfgs'):
            
            covMat = maxRslt['covMat']
            
        elif(hessian == 'numdiff'):
            
            critFunc = maxCls.getAttr('critFunc')
            
            ndObj    = nd.Hessian(lambda x: clsMax._scipyWrapperFunction(x, critFunc)) 
            hess     = ndObj(xopt)
            covMat   = np.linalg.pinv(hess)       

        pkl.dump(covMat, open('covMat.grm.pkl', 'wb'))
                 
    # Construct result class.
    rslt = clsRslt.results()

    rslt.setAttr('grmObj', grmObj)   
        
    rslt.setAttr('maxRslt', maxRslt)    
    
    rslt.setAttr('covMat', covMat)    

    rslt.lock()

    rslt.store('rsltObj.grm.pkl')

    return rslt