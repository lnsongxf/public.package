''' Module contains all functions required to generate a model object
    from the processed initialization file.
'''

# standard library
import numpy as np

# project library
from grmpy.clsModel import modelCls
from grmpy.tools.msc import *


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
    
    rslt    = create_matrices(dataset, initDict)

    # Initialize model object.
    modelObj = modelCls()
    
    modelObj.set_attr('Y', rslt['Y'])
    
    modelObj.set_attr('D', rslt['D'])
    
    
    modelObj.set_attr('x_ex_post', rslt['x_ex_post'])
    
    modelObj.set_attr('x_ex_ante', rslt['x_ex_ante'])
        
    
    modelObj.set_attr('G', rslt['G'])
    
    modelObj.set_attr('Z', rslt['Z'])
    
    
    modelObj.set_attr('numCovarsExclBeneExPost', numCovarsExclBeneExPost)
    
    modelObj.set_attr('numCovarsExclBeneExAnte', numCovarsExclBeneExAnte)
    
    modelObj.set_attr('numCovarsExclCost', numCovarsExclCost)


    modelObj.set_attr('algorithm', initDict['ESTIMATION']['algorithm'])

    modelObj.set_attr('epsilon', initDict['ESTIMATION']['epsilon'])

    modelObj.set_attr('differences', initDict['ESTIMATION']['differences'])

    modelObj.set_attr('gtol', initDict['ESTIMATION']['gtol'])

    modelObj.set_attr('maxiter', initDict['ESTIMATION']['maxiter'])

    modelObj.set_attr('withAsymptotics', initDict['ESTIMATION']['asymptotics'])

    modelObj.set_attr('numDraws', initDict['ESTIMATION']['draws'])

    modelObj.set_attr('version', initDict['ESTIMATION']['version'])

    modelObj.set_attr('hessian', initDict['ESTIMATION']['hessian'])

    modelObj.set_attr('alpha', initDict['ESTIMATION']['alpha'])


    
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