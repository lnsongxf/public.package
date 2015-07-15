''' Interface for all routines related to the processing of the 
    initialization file.
'''

# standard library
import os

# project library
import grmpy.user._createDictionary as auxDict
import grmpy.user._createModel      as auxModel
import grmpy.user._createParas      as auxParas
import grmpy.user._createRequest    as auxRequest

''' Main function.
'''
def initialize(initFile, use_simulation = False, is_simulation = False):
    ''' Read initialization file and construct the objects required for the 
        estimation runs.
    '''
    # Antibugging.
    assert (os.path.exists(initFile))
    
    # Process initialization file.
    initDict = auxDict.processInput(initFile)
    
    # Use SIMULATION info.
    if(use_simulation):
        
        initDict['DATA']['source'] = initDict['SIMULATION']['target']
        
        initDict['DATA']['agents'] = initDict['SIMULATION']['agents']
    
    # Construct objects.
    modelObj   = auxModel.constructModel(initDict)
    
    parasObj   = auxParas.constructParas(initDict, modelObj, is_simulation)

    # Quality checks.
    for obj in [modelObj, parasObj]:

        assert (obj.getStatus() == True)
    
    # Finishing
    return modelObj, parasObj, initDict