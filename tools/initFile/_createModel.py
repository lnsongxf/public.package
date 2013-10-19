''' Module contains all functions required to generate a model object
    from the processed initialization file.
'''

# standard library
import numpy as np

# project library
import grmToolbox

''' Main function.
'''
def constructModel(initDict):
    ''' Create  model object based on dictionary that contains the information
        from the initialization file.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Distribute initialization file.
    numCovarsExclBeneExAnte = initDict['DERIV']['exclBene']['exAnte']['num']

    numCovarsExclBeneExPost = initDict['DERIV']['exclBene']['exPost']['num']
    
    numCovarsExclCost       = initDict['DERIV']['exclCost']['num']
    
    # Construct data array.
    dataset = _processDataset(initDict)
    
    rslt    = grmToolbox.createMatrices(dataset, initDict)

    # Initialize model object.
    modelObj = grmToolbox.modelCls()
    
    
    modelObj.setAttr('Y', rslt['Y'])
    
    modelObj.setAttr('D', rslt['D'])
    
    
    modelObj.setAttr('xExPost', rslt['xExPost'])
    
    modelObj.setAttr('xExAnte', rslt['xExAnte'])
        
    
    modelObj.setAttr('G', rslt['G'])
    
    modelObj.setAttr('Z', rslt['Z'])
    
    
    modelObj.setAttr('numCovarsExclBeneExPost', numCovarsExclBeneExPost)
    
    modelObj.setAttr('numCovarsExclBeneExAnte', numCovarsExclBeneExAnte)
    
    modelObj.setAttr('numCovarsExclCost', numCovarsExclCost)
    
    
    modelObj.lock()
    
    # Finishing.
    return modelObj

''' Private auxiliary functions.
'''
def _processDataset(initDict):
    ''' Processing of dataset by removing missing variables and subset.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
   
    # Distribute initialization file.
    fileName = initDict['DATA']['source']
    
    numObs   = initDict['DATA']['agents']

    allPos   = initDict['DERIV']['pos']['all']

    # Load source.   
    dataset = np.genfromtxt(fileName)

    # Restrict to non-missing.
    idx = []
    
    for i in range(dataset.shape[0]):
        
        noMissings = np.all(np.isfinite(dataset[i,allPos]))
        
        if(noMissings):
            
            idx.append(i)
    
    dataset = dataset[idx, :]
   
    # Subset selection.
    if(numObs is not None):
        
        dataset = dataset[:numObs,:]

    # Quality checks.
    assert (np.all(np.isfinite(dataset[:,allPos])))
        
    # Finishing.    
    return dataset