#!/usr/bin/env python
''' Perturb starting parameter values.
'''
# standard library
import numpy   as np

import sys
import os
import argparse

# Pythonpath
dir_ = os.path.dirname(os.path.realpath(__file__)).replace('/scripts', '')
sys.path.insert(0, dir_)

# project library
import grmToolbox


''' Auxiliary function.
'''
def process(args):
    ''' Process arguments.
    '''
    # Distribute arguments.
    seed, scale, init = args.seed, args.scale, args.init
    
    # Quality checks.
    assert (os.path.exists(init))
    assert (isinstance(seed, int))
    assert (isinstance(scale, float))
    assert (scale >= 0)
    
    # Finishing.
    return seed, scale, init

''' Main function.
'''
def perturb(scale = 0.1, seed = 123, init = 'init.ini'):
    ''' Perturb current values of structural parameters.
    '''
    #Process initialization file.
    _, parasObj, _, _ = grmToolbox.initialize(init, useSimulation = False)
    
    ''' Update parameter object.
    '''
    # Antibugging.
    assert (os.path.isfile('stepParas.grm.out'))
    
    # Update parameter objects.
    parasObj = grmToolbox.updateParameters(parasObj)
    
    parasObj.updateStart()
            
    ''' Perturb external values.
    '''
    np.random.seed(seed)
        
    baseValues = parasObj.getValues('external', 'free')
       
    perturb    = (np.random.sample(len(baseValues)) - 0.5)*scale
        
    evalPoints = baseValues + perturb
        
    ''' Transform evaluation points.
    '''
    parasObj.update(evalPoints, 'external', 'free')
        
    evalPoints = parasObj.getValues('internal', 'all')
        
    ''' Finishing.
    '''
    np.savetxt('stepParas.grm.out',  evalPoints, fmt = '%15.10f')
        
''' Execution of module as script.
'''
if __name__ == '__main__':
    
    parser  = argparse.ArgumentParser(description = \
                'Perturb current value of structural parameters.')
    
    parser.add_argument('--seed', \
                        type    = int , \
                        default = 123, \
                        dest    = 'seed', \
                        help    = 'value of random seed')
    
    parser.add_argument('--scale', \
                        type    = float , \
                        default = 0.1, \
                        dest    = 'scale', \
                        help    = 'magnitude of perturbation')
    
    parser.add_argument('--init', \
                        action  = 'store', \
                        dest    = 'init', \
                        default = 'init.ini', \
                        help    = 'source for model configuration')
    
    
    args              = parser.parse_args()
    
    seed, scale, init = process(args)
    
    
    perturb(scale = scale, seed = seed, init = init)