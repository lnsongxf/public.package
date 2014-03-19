#!/usr/bin/env python
''' Module for unit tests related to the parameter management and updating. 
'''
# standard library
import os
import sys

import cPickle as pkl

from nose.core   import *
from nose.tools  import *

# set working directory
dir_ = os.path.abspath(os.path.split(sys.argv[0])[0])
os.chdir(dir_)

# Pythonpath
dir_ = os.path.dirname(os.path.realpath(__file__)).replace('/tests', '')
sys.path.insert(0, dir_)

import grmToolbox

from tools.workflow.estimation import estimate

''' Test class.
'''
class testParasCls(object):
    ''' Test class.
    '''
    def testA(self):
        ''' Test parameter transformations.
        '''
        # Run command.
        initFile = '../dat/testInit_A.ini'
        
        estimate(initFile, restart = False, useSimulation = False)
        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
                
        # Access class attributes
        parasObj = rsltObj.getAttr('parasObj')
        
        paraObjs = parasObj.getAttr('paraObjs')
        
        for paraObj in paraObjs:
            
            value = paraObj.getAttr('value')
            
            ext   = parasObj._transformToExternal(paraObj, value)
            
            int_  = parasObj._transformToInternal(paraObj, ext)
            
            assert_almost_equal(value, int_)
                    
        # Cleanup.
        grmToolbox.cleanup(isRestart = False)
        
if __name__ == '__main__': 
    
    runmodule()   
