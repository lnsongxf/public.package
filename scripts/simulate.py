#!/usr/bin/env python
''' Script to simulate a sample for estimation with the grmToolbox.
'''

# standard library
import os
import sys
import random
import argparse

import numpy    as np

# project library
dir_ = os.path.realpath(__file__).replace('/scripts/simulate.py','')
sys.path.insert(0, dir_)

import grmToolbox

''' Main function.
'''
def simulate(init = 'init.ini', update = False):
    ''' Simulate dataset for grmToolbox.
    '''
    
    isMock = _createMock(init)
    
    ''' Process initialization file.
    '''
    _, parasObj, _, initDict = grmToolbox.initialize(init, isSimulation = True)
    
    ''' Distribute information.
    '''
    target = initDict['SIMULATION']['target']
    
    seed   = initDict['SIMULATION']['seed']
    
    np.random.seed(seed); random.seed(seed)

    ''' Update parameter class.
    '''
    if(update): parasObj = grmToolbox.updateParameters(parasObj)
    
    ''' Create simulated dataset.
    '''   
    if(isMock): os.remove(initDict['DATA']['source'])
        
    simAgents   = initDict['SIMULATION']['agents']
    
    max_        = initDict['DERIV']['pos']['max']
    
    simDat      = np.empty((simAgents, max_ + 1), dtype = 'float')
    
    simDat[:,:] = np.nan
        
    
    simDat = _simulateExogenous(simDat, initDict)
    
    simDat = _simulateEndogenous(simDat, parasObj, initDict)
    
    ''' Update for prediction step.
    '''
    rslt = grmToolbox.createMatrices(simDat, initDict)
    
    parasObj.unlock()
    
    parasObj.setAttr('xExAnte', rslt['xExAnte'])
    
    parasObj.setAttr('xExPost', rslt['xExPost'])
    
    parasObj.lock()
    
    ''' Save dataset. 
    '''
    np.savetxt(target, simDat, fmt = '%15.10f')
    
    likl = _getLikelihood(init)
    
    _writeInfo(parasObj, target, rslt, likl)

''' Auxiliary functions.
'''
def _getLikelihood(init):
    ''' Calculate likelihood for simulated dataset at true parameter values.
    '''
    # Antibugging.
    assert (isinstance(init, str))
    
    # Process model ingredients.
    modelObj, parasObj, requestObj, _ = grmToolbox.initialize(init, True)
    
    # Initialize container.
    grmObj = grmToolbox.grmCls()
    
    grmObj.setAttr('modelObj', modelObj)
    
    grmObj.setAttr('requestObj', requestObj)

    grmObj.setAttr('parasObj', parasObj)
    
    grmObj.lock()


    critObj = grmToolbox.critCls(grmObj)
    
    critObj.lock()
    
    # Evaluate at true values. 
    x    = parasObj.getValues('external', 'free')
    
    likl = grmToolbox.evaluate(x, critObj)
    
    # Cleanup.
    for file_ in ['grmToolbox.grm.log', 'stepParas.grm.out', \
                  'startParas.grm.out']: 
        
        os.remove(file_)
    
    # Finishing.
    return likl
    
def _createMock(init):
    ''' Create a mock dataset which allows for use of existing routines 
        in the case of a missing source dataset. 
    '''
    
    initDict = grmToolbox.processInput(init)
    
    isMock = (os.path.exists(initDict['DATA']['source']) == False)
    
    obsAgents = initDict['DATA']['agents']
    
    pos  = initDict['DATA']['treatment']
    
    max_      = initDict['DERIV']['pos']['max']
    
    simDat      = np.empty((obsAgents, max_ + 1), dtype = 'float')
    
    simDat[:,:] = np.random.randn(obsAgents, max_ + 1)
    
    simDat[:,pos] = np.random.random_integers(0, 1, obsAgents)
    
    source = initDict['DATA']['source']
    
    np.savetxt(source, simDat, fmt = '%15.10f')
    
    # Finishing.
    return isMock

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

    all_      = initDict['DERIV']['pos']['all']

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

def _distributeInput(parser):
    ''' Check input for estimation script.
    '''
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    init   = args.init 
    update = args.update 

    # Assertions.
    assert (init is not None)
    assert (os.path.exists(init))
    assert (update in [True, False])
    
    # Finishing.
    return init, update

def _writeInfo(parasObj, target, rslt, likl):
    ''' Write out some additional infos about the simulated dataset.
    '''
    
    # Auxiliary objects.
    fileName     = target.split('.')[0] 
    
    numAgents    = str(len(rslt['Y']))
    
    numTreated   = np.sum(rslt['D'] == 1)
    
    numUntreated = np.sum(rslt['D'] == 0)
    
    fval         = str(likl)    
    
    # Write out structural parameters.
    paras = parasObj.getValues(version = 'internal', which = 'all')
    
    np.savetxt(fileName + '.paras.grm.out', paras, fmt = '%15.10f')
    
    # Write out information on agent experiences.    
    with open(fileName + '.infos.grm.out', 'w') as file_:
         
        file_.write('\n Simulated Economy\n\n')
        
        file_.write('   Number of Observations: ' + numAgents + '\n')

        file_.write('   Function Value:         ' + fval + '\n\n')

        string  = '''{0[0]:<10} {0[1]:>12}\n'''
    
        file_.write('   Choices:  \n\n') 
        
        file_.write(string.format(['     Treated  ', numTreated]))

        file_.write(string.format(['     Untreated', numUntreated]))

        file_.write('\n\n')    
        
        
        string  = '''{0[0]:<10} {0[1]:15.5f}\n'''
        
        file_.write('   Outcomes:  \n\n') 
        
        file_.write(string.format(['     Treated  ', np.mean(rslt['Y'][rslt['D'] == 1])]))

        file_.write(string.format(['     Untreated', np.mean(rslt['Y'][rslt['D'] == 0])]))
        
        
        file_.write('\n\n')
            
''' Execution of module as script.
'''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 
        'Start simulation using the grmToolbox.', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--init', \
                        action  = 'store', \
                        dest    = 'init', \
                        default = 'init.ini', \
                        help    = 'source for model configuration')
    
    parser.add_argument('--update', \
                        action  = 'store_true', \
                        dest    = 'update', \
                        default = False, \
                        help    = 'update structural parameters')
    
    init, update = _distributeInput(parser)
    
    simulate(init = init, update = update)
        