#!/usr/bin/env python
''' Script for the estimation.
'''
# standard library
import os
import sys
import argparse 

# project library
dir_ = os.path.realpath(__file__).replace('scripts/clean.py','')
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
    resume = args.resume

    # Assertions.
    assert (resume in [True, False])    

    # Finishing.
    return resume

''' Execution of module as script.
'''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 
        'Cleanup after an estimation run of the grmToolbox.', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--resume', \
                        action  = 'store_true', \
                        dest    = 'resume', \
                        default = False, \
                        help    = 'keep files required to resume estimation')
    
    resume = _distributeInput(parser)
    
    grmToolbox.cleanup(resume)