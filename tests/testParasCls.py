#!/usr/bin/env python
''' Module for unit tests related to the parameter management and updating. 
'''
# standard library
import os
import sys

import cPickle as pkl


# module variables
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.getcwd()

# testing library
from nose.core import runmodule
from nose.tools import *

# Pythonpath
dir_ = FILE_PATH.replace('/tests', '')
sys.path.insert(0, dir_)

# project library
import interface as grmToolbox
from scripts.estimate import estimate


''' Test class.
'''
class testParasCls(object):
    ''' Test class.
    '''

    def setup(self):

        os.chdir(FILE_PATH)

    def teardown(self):

        os.chdir(TEST_PATH)

    def testA(self):
        ''' Test parameter transformations.
        '''

        #grmToolbox.cleanup(resume = False)

        # Run command.
        initFile = '../dat/testInit_A.ini'
        
        estimate(initFile, resume = False, useSimulation = False)
        
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
        grmToolbox.cleanup(resume = False)
        
if __name__ == '__main__': 
    
    runmodule()   
