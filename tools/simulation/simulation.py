#!/usr/bin/env python
''' Script to simulate a sample for estimation with the grmToolbox.
'''

# standard library
import os
import sys
import argparse

import numpy    as np

# project library
dir_ = os.path.realpath(__file__).replace('/tools/simulation/simulation.py','')
sys.path.insert(0, dir_)

import grmToolbox
import _auxiliaryFunctions as auxFunc


''' Auxiliary functions.
'''
def _distributeInput(parser):
    ''' Check input for estimation script.
    '''
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    initFile = args.initFile 

    # Assertions.
    assert (initFile is not None)
    assert (os.path.exists(initFile))
    
    # Finishing.
    return initFile

''' Process command line arguments.
'''
parser = argparse.ArgumentParser(description = 
'Start simulation using the grmToolbox.')

parser.add_argument('-init', \
                    action  = 'store', \
                    dest    = 'initFile', \
                    default = None, \
                    help    = 'Initialization file.')

initFile = _distributeInput(parser)

''' Process initialization file.
'''
_, parasObj, _, initDict = grmToolbox.initialize(initFile)

''' Distribute information.
'''
target = initDict['SIMULATION']['target']

seed   = initDict['SIMULATION']['seed']

np.random.seed(seed)

''' Update parameter class.
'''
parasObj = grmToolbox.updateParameters(parasObj)

''' Create simulated dataset.
'''
simAgents = initDict['SIMULATION']['agents']

max_      = initDict['DERIV']['pos']['max']

simDat      = np.empty((simAgents, max_ + 1), dtype = 'float')

simDat[:,:] = np.nan
    

simDat = auxFunc._simulateExogenous(simDat, initDict)

simDat = auxFunc._simulateEndogenous(simDat, parasObj, initDict)

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