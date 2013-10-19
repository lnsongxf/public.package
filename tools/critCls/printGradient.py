#!/usr/bin/env python
''' Script to print gradient in the grmToolbox.
'''

# standard library
import os
import sys
import argparse

from time import localtime, strftime

import cPickle  as pkl
import numpy    as np

# edit pythonpath
dir_ = os.path.realpath(__file__).replace('/tools/critCls/printGradient.py','')
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
    initFile = args.initFile 

    # Assertions.
    assert (initFile is not None)
    assert (os.path.exists(initFile))
    
    # Finishing.
    return initFile

''' Process command line arguments.
'''
parser = argparse.ArgumentParser(description = 
'Print gradient at current evaluation point.')

parser.add_argument('-init', \
                    action  = 'store', \
                    dest    = 'initFile', \
                    default = None, \
                    help    = 'Initialization file.')

initFile = _distributeInput(parser)

''' Load ingredients.
'''
modelObj, parasObj, requestObj, _ = grmToolbox.initialize(initFile)

''' Update parameter class.
'''
parasObj = grmToolbox.updateParameters(parasObj)

''' Prepare interface.
'''    
# Toolbox object.
grmObj = grmToolbox.grmCls()
    
grmObj.setAttr('modelObj', modelObj)
    
grmObj.setAttr('requestObj', requestObj)

grmObj.setAttr('parasObj', parasObj)
    
grmObj.lock()

# Criterion object.
critFunc = grmToolbox.critCls(grmObj)
        
critFunc.lock()            

''' Get information.
'''
externalValues = parasObj.getValues(isExternal = True, isAll = False)    

gradientValues = critFunc.evaluate(externalValues, 'gradient')

functionValue  = critFunc.evaluate(externalValues, 'function')

timeStamp      = strftime("%Y-%m-%d %H:%M:%S", localtime())

criterionValue = np.amax(np.abs(gradientValues))

''' Write out information.
'''
file_ = open('gradientInfo.grm.out', 'w')

file_.write('\n' + ' ' + timeStamp + '\n\n')

file_.write(' Function Value ' + '\n\n')

file_.write('{0:20.15f}'.format(functionValue) + '\n\n')

file_.write(' Gradient ' + '\n\n')

for grad in gradientValues:
    
    str_ = '{0:20.15f}'.format(grad) + '\n'
    
    file_.write(str_)

file_.write('\n' + ' Convergence Criterion ' + '\n\n')

file_.write('{0:20.15f}'.format(criterionValue) + '\n\n')

file_.close()