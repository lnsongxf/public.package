#!/usr/bin/env python
''' Module to ease modification of parasObj along the path.
    
        Help:
        
            modifyParameters -h 

'''
# standard library
import os
import sys
import argparse 

import cPickle      as pkl
import numpy        as np

# edit pythonpath
dir_ = os.path.realpath(__file__).replace('/tools/parasCls/modifyParameters.py','')
sys.path.insert(0, dir_)

# project library
import clsParas

''' Process command line arguments.
'''
parser = argparse.ArgumentParser(description = 
'Modify parameters of an instance of the paraCls.')

parser.add_argument('-operation', \
                    action  = 'store', \
                    dest    = 'operation', \
                    default = None, \
                    help    = 'Operation to perform.')

parser.add_argument('-bounds', \
                    action  = 'store', \
                    dest    = 'bounds', \
                    default = None, \
                    nargs = '+', \
                    help    = 'New bounds.')

parser.add_argument('-value', \
                    action  = 'store', \
                    dest    = 'value', \
                    default = None, \
                    help    = 'Set value.')

parser.add_argument('-count', \
                    action  = 'store', \
                    dest    = 'count', \
                    help    = 'Identify parameter by count.')

args = parser.parse_args()

''' Check arguments.
'''
# Operation.
operation = args.operation

assert (operation in ['modifyValue', 'modifyBounds', 'fix', 'free'])

# Bounds.
bounds = args.bounds

if(bounds is not None):
    
    bounds = list(bounds)
    
    # Type conversion
    assert (len(bounds) == 2)
    
    for i in range(2):
        
        if(bounds[i] != 'None'):
            
            bounds[i] = float(bounds[i])
        
        else:
            
            bounds[i] = None
    
    bounds = tuple(bounds)
    
# Value.

value = args.value

if(value is not None):
    
    value = float(value)

# Process count.

try:
    
    counts = int(args.count)

except ValueError:
    
    lower = int(args.count.split('-')[0])
    upper = int(args.count.split('-')[1])

    counts = range(lower, (upper + 1))

''' Check presence of required file.
'''
parasObj = pkl.load(open('parasObj.grm.pkl', 'r'))

''' Update starting objects with most recent external parameters. 
'''
hasStep = (os.path.isfile('stepParas.grm.out'))

if(hasStep):
    
    internalValues = np.array(np.genfromtxt('stepParas.grm.out'), dtype = 'float', ndmin = 1)

    parasObj.updateValues(internalValues, isExternal = False, isAll = False)       

''' Operations.
'''
parasObj = parasObj.modifyParameter(counts, operation, bounds, value)

parasObj.updateStart()

''' Finalizing.
'''
# Write out current values. 
paras = parasObj.getValues(isExternal = False, isAll = False)

np.savetxt('stepParas.grm.out',  paras, fmt = '%12.7f')

# Save file.
parasObj.store('parasObj.grm.pkl')
