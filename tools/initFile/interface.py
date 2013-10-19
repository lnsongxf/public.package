''' Interface for all routines related to the processing of the 
    initialization file.
'''

# standard library
import os

# project library
import _createDictionary as auxDict
import _createModel      as auxModel
import _createParas      as auxParas
import _createRequest    as auxRequest

''' Main function.
'''
def initialize(initFile, useSimulation = False):
    ''' Read initialization file and construct the objects required for the 
        estimation runs.
    '''
    # Antibugging.
    assert (os.path.exists(initFile))
    
    # Process initialization file.
    initDict = auxDict.processInput(initFile)
    
    # Use SIMULATION info.
    if(useSimulation):
        
        initDict['DATA']['source'] = initDict['SIMULATION']['target']
        
        initDict['DATA']['agents'] = initDict['SIMULATION']['agents']
    
    # Construct objects.
    modelObj   = auxModel.constructModel(initDict)
    
    parasObj   = auxParas.constructParas(initDict, modelObj)
    
    requestObj = auxRequest.constructRequest(initDict)
    
    # Quality checks.
    for obj in [modelObj, parasObj, requestObj]:

        assert (obj.getStatus() == True)
    
    # Finishing
    return modelObj, parasObj, requestObj, initDict