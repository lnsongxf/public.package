''' Module for auxiliary functions that are used throughout the grmToolbox.
'''
# standard library
import shutil
import glob
import os

import numpy            as np

# project library
import interface as grmToolbox


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

def updateParameters(parasObj):
    ''' Update parameter object if possible.
    '''
    # Antibugging.
    assert (isinstance(parasObj, grmToolbox.parasCls))
    assert (parasObj.getStatus() == True)
    
    # Update.
    hasStep = (os.path.isfile('stepParas.grm.out'))
    
    if(hasStep):
        
        internalValues = np.array(np.genfromtxt('stepParas.grm.out'), dtype = 'float', ndmin = 1)
    
        parasObj.update(internalValues, version = 'internal', which = 'all')
    
    # Finishing.
    return parasObj

def cleanup(resume):
    ''' Cleanup from previous estimation run.
    '''
    # Antibugging.
    assert (resume in [True, False])
    
    # Construct files list.
    fileList = glob.glob('*.grm.*')
        
    if(resume): 
        
        for file_ in ['stepParas.grm.out']:
            
            try:
                
                fileList.remove(file_)
    
            except:
                
                pass
    
    # Remove information from simulated data.
    for file_ in ['*.infos.grm.out', '*.paras.grm.out']:
                        
        try:
            
            fileList.remove(glob.glob(file_)[0])
            
        except:
            
            pass
    
    # Cleanup
    if(os.path.exists('grm.rslt')): shutil.rmtree('grm.rslt')
               
    for file_ in fileList:
        
        try:
            
            os.remove(file_)
            
        except OSError:
            
            pass