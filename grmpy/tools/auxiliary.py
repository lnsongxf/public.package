""" Module for auxiliary functions that are used throughout the grmToolbox.
"""

# standard library
import numpy as np
import shlex
import os


def _updateParameters(parasObj):
    ''' Update parameter object if possible.
    '''
    # Antibugging.
    assert (parasObj.get_status() == True)
    assert (os.path.isfile('info.grmpy.out'))

    # Update.
    hasStep = (os.path.isfile('info.grmpy.out'))

    if(hasStep):

        list_ = []

        is_relevant = False

        with open('info.grmpy.out', 'r') as file_:

            for line in file_:

                currentLine = shlex.split(line)

                if len(currentLine) ==  0:
                    continue

                if len(currentLine) > 1:
                    break

                if currentLine == ['STOP']:
                    is_relevant = True

                if currentLine[0] in ['START', 'STOP']:
                    continue

                if is_relevant:
                    list_ += [np.float_(currentLine[0])]

        starting_values = np.array(list_)

        parasObj.update(starting_values, version = 'internal', which = 'all')

    # Finishing.
    return parasObj

def createMatrices(dataset, initDict):
    ''' Create the data matrices.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    
    assert (isinstance(dataset, np.ndarray))
    assert (dataset.dtype == 'float')
    assert (dataset.ndim == 2)
    
    # Distribute information
    outcome   = initDict['DATA']['outcome']

    treatment = initDict['DATA']['treatment'] 

    common    = initDict['DERIV']['common']['pos']

    exclBeneExAnte = initDict['DERIV']['exclBene']['exAnte']['pos']

    exclBeneExPost = initDict['DERIV']['exclBene']['exPost']['pos']

    exclCost       = initDict['DERIV']['exclCost']['pos']

    # Construct auxiliary information.
    numAgents = dataset.shape[0]
    
    # Create matrices.
    Y = dataset[:,outcome]
    
    D = dataset[:,treatment]
    
    M = dataset[:,common].copy()
        
    M = np.concatenate((M, np.ones((numAgents, 1))), axis = 1)
        
        
    xExAnte = np.concatenate((dataset[:,exclBeneExAnte], M), axis = 1)
    
    xExPost = np.concatenate((dataset[:,exclBeneExPost], M), axis = 1)
        
        
    G = np.concatenate((M, dataset[:, exclCost]), axis = 1) 
        
    Z = np.concatenate((xExAnte, dataset[:, exclCost]), axis = 1)
    
    # Quality checks.
    for mat in [xExAnte, xExPost, G, Z]:
        
        assert (isinstance(mat, np.ndarray))
        assert (mat.dtype == 'float')
        assert (mat.ndim == 2)
    
    for mat in [D, Y]:
        
        assert (isinstance(mat, np.ndarray))
        assert (mat.dtype == 'float')
        assert (mat.ndim == 1)

    # Collect.
    rslt = {}
    
    rslt['xExPost'] = xExPost

    rslt['xExAnte'] = xExAnte
    
    
    rslt['G'] = G
    
    rslt['Z'] = Z
    
    
    rslt['Y'] = Y
        
    rslt['D'] = D
        
    # Finishing.
    return rslt

