''' Module with auxiliary functions related to the dataset simulation.
'''

# standard library
import os

import numpy as np

# project library
import grmToolbox

def _simulateEndogenous(simDat, parasObj, initDict):
    ''' Simulate the endogenous characteristics such as choices and outcomes.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(simDat, np.ndarray))
    assert (isinstance(parasObj, grmToolbox.parasCls))
    assert (parasObj.getStatus() == True)
    
    # Distribute information.
    simAgents = initDict['SIMULATION']['agents']
    
    outcome   = initDict['DATA']['outcome'] 
    
    treatment = initDict['DATA']['treatment'] 
    
    all_      = initDict['DERIV']['pos']['all']
        
    # Sampling of unobservables.
    varV  = parasObj.getParameters('var',  'V')  
    
    varU1 = parasObj.getParameters('var',  'U1') 
    
    varU0 = parasObj.getParameters('var',  'U0') 
    
    mean   = np.tile(0.0, 3)
    
    covMat = np.diag([varU1, varU0, varV])
    
    covMat[2,0] = covMat[0,2] = parasObj.getParameters('cov', 'U1,V') 
    
    covMat[2,1] = covMat[1,2] = parasObj.getParameters('cov', 'U0,V') 
    
    U1, U0, V = np.random.multivariate_normal(mean, covMat, simAgents).T
    
    # Create data matrices.
    rslt    = grmToolbox.createMatrices(simDat, initDict)
    
    xExPost = rslt['xExPost'] 
    
    Z       = rslt['Z'] 

    # Simulate choices.
    coeffsChoc = parasObj.getParameters('choice', None)
    
    D = (np.dot(coeffsChoc, Z.T) - V > 0.0)
    
    # Potential Outcomes
    outcTreated   = parasObj.getParameters('outc', 'treated')
    outcUntreated = parasObj.getParameters('outc', 'untreated') 
            
    Y1 = np.dot(outcTreated, xExPost.T)   + U1
    Y0 = np.dot(outcUntreated, xExPost.T) + U0
    
    Y = D*Y1 + (1 - D)*Y0
        
    simDat[:,outcome]   = Y
    simDat[:,treatment] = D
      
    # Quality checks.
    assert (isinstance(simDat, np.ndarray))
    assert (np.all(np.isfinite(simDat[:,all_])))
    assert (simDat.dtype == 'float')

    # Finishing.
    return simDat

def _simulateExogenous(simDat, initDict):
    ''' Simulate the exogenous characteristics by filling up the data frame
        with random deviates of the exogenous characteristics from the
        observed dataset.
    '''
    # Antibugging.
    assert (isinstance(initDict, dict))
    assert (isinstance(simDat, np.ndarray))
    
    # Distribute information.
    source    = initDict['DATA']['source']

    simAgents = initDict['SIMULATION']['agents']

    all_      =  initDict['DERIV']['pos']['all']

    outcome   = initDict['DATA']['outcome']
    
    treatment = initDict['DATA']['treatment']
    
    # Restrict to exogenous positions.
    for pos in [outcome, treatment]:
        
        all_.remove(pos)
    
    # Simulate endogenous characteristics.
    hasSource   = (os.path.exists(source) == True)
    
    if(hasSource):
        
        obsDat    = np.genfromtxt(source)
        
        obsAgents = obsDat.shape[0]
        
        if(obsAgents == simAgents):
            
            idx_ = range(obsAgents)
            
        else:
            
            idx_ = np.random.randint(0, obsAgents, size = simAgents)
        
        
        for pos in all_:
            
            simDat[:,pos] = obsDat[idx_,pos]
        
    else:
        
        for pos in all_:
            
            simDat[:,pos] = np.random.randn(simAgents) 
    
    # Quality checks.
    assert (isinstance(simDat, np.ndarray))
    assert (np.all(np.isfinite(simDat[:,all_])))
    assert (simDat.dtype == 'float')

    # Finishing.
    return simDat
