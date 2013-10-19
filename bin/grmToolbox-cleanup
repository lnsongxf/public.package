#!/usr/bin/env python
''' Script for the estimation.
'''
# standard library
import os
import sys
import argparse 

# project library
dir_ = os.path.realpath(__file__).replace('/tools/workflow/cleanup.py','')
sys.path.insert(0, dir_)

import grmToolbox

''' Process command line arguments.
'''
parser = argparse.ArgumentParser(description = 
'Cleanup after an estimation run of the grmToolbox.')

parser.add_argument('-restart', \
                    action  = 'store', \
                    dest    = 'isRestart', \
                    default = False, \
                    help    = 'Maintain restart files.')

args = parser.parse_args()

isRestart = args.isRestart

isRestart = bool(isRestart)

grmToolbox.cleanup(isRestart)