''' Module that constructs the request object from the initialization file.
'''

# project library
from grmpy.clsRequest import requestCls

''' Construct request object.
'''
def constructRequest(initDict):
    
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Initialize request.    
    requestObj = requestCls()
    
    requestObj.setAttr('algorithm', initDict['ESTIMATION']['algorithm'])
    

    requestObj.setAttr('epsilon', initDict['ESTIMATION']['epsilon'])
    
    requestObj.setAttr('differences', initDict['ESTIMATION']['differences'])
    
    
    requestObj.setAttr('gtol', initDict['ESTIMATION']['gtol'])
    
    requestObj.setAttr('maxiter', initDict['ESTIMATION']['maxiter'])
     
    requestObj.setAttr('withAsymptotics', initDict['ESTIMATION']['asymptotics'])
    
    requestObj.setAttr('numDraws', initDict['ESTIMATION']['draws'])

    requestObj.setAttr('version', initDict['ESTIMATION']['version'])

    requestObj.setAttr('hessian', initDict['ESTIMATION']['hessian'])

    requestObj.setAttr('alpha', initDict['ESTIMATION']['alpha'])

    requestObj.lock()

    # Finishing.
    return requestObj