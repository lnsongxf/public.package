#!/usr/bin/env python
''' Script for the estimation.
'''
# standard library
import os
import sys
import argparse

import cPickle as pkl
import numpy   as np

# project library
dir_ = os.path.realpath(__file__).replace('/tools/workflow/estimation.py','')
sys.path.insert(0, dir_)

import grmToolbox

''' Auxiliary functions.
'''
def _distributeInput(parser):
    ''' Check input for estimation script.
    '''
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    restart       = args.restart
    initFile      = args.initFile 
    useSimulation = args.simulation

    # Assertions.
    assert (restart in [True, False])
    assert (useSimulation in [True, False])
    assert (initFile is not None)
    assert (os.path.exists(initFile))
    
    # Finishing.
    return initFile, restart, useSimulation

def fork():
    ''' Fork child process to run estimation in the background.
    '''
        
    pid = os.fork()

    if(pid > 0): sys.exit(0)

    pid = os.getpid()
    
    np.savetxt('.pid', [pid], fmt ='%d')
    
''' Process command line arguments.
'''
def estimate(initFile, restart, useSimulation):
    ''' Estimate specified model.
    '''
    
    # Cleanup
    grmToolbox.cleanup(restart)
    
    #Process initialization file.
    modelObj, parasObj, requestObj, _ = grmToolbox.initialize(initFile, useSimulation)
    
    # Process restart.
    if(restart):
        
        # Antibugging.
        assert (os.path.isfile('parasObj.grm.pkl'))
        assert (os.path.isfile('stepParas.grm.out'))
        
        # Load parameter objects.   
        parasObj = pkl.load(open('parasObj.grm.pkl', 'r'))
        
        # Integrity checks. 
        modelAgents = modelObj.getAttr('numAgents')
        
        parasAgents = parasObj.getAttr('numAgents')
    
        assert (modelAgents == parasAgents)    
    
        # Update parameter objects.
        parasObj = grmToolbox.updateParameters(parasObj)
    
        parasObj.updateStart()
        
        parasObj.store('parasObj.grm.pkl')
    
    else:
        
        parasObj.store('parasObj.grm.pkl')
       
        paras = parasObj.getValues(isExternal = False, isAll = False)
            
        np.savetxt('stepParas.grm.out', paras, fmt = '%25.12f')
            
    # Run optimization.
    grmToolbox.maximize(modelObj, parasObj, requestObj)

''' Execution of module as script.
'''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 
    'Start estimation run using the grmToolbox.')
    
    parser.add_argument('-restart', \
                        action  = 'store_true', \
                        dest    = 'restart', \
                        default = False, \
                        help    = 'Restart estimation run.')
    
    parser.add_argument('-simulation', \
                        action  = 'store_true', \
                        dest    = 'simulation', \
                        default = False, \
                        help    = 'Use SIMULATION information.')
    
    parser.add_argument('-init', \
                        action  = 'store', \
                        dest    = 'initFile', \
                        default = None, \
                        help    = 'Initialization file.')
    
    
    fork() 

    initFile, restart, useSimulation = _distributeInput(parser)

    estimate(initFile, restart, useSimulation)
