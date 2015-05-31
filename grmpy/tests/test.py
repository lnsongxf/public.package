#!/usr/bin/env python
""" Module for unit tests related to the parameter management and updating.
"""
# standard library
import os
import sys

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
from grmpy.tools.auxiliary import cleanup
from grmpy.user._createParas import constructParas
from grmpy.user._createDictionary import  processInput
from grmpy.user._createModel import constructModel

''' Test class.
'''
class testParasCls(object):
    """ Test class.
    """
    def setup(self):

        os.chdir(FILE_PATH)

    def teardown(self):

        os.chdir(TEST_PATH)

    def testA(self):
        """ Test parameter transformations.
        """
        init_dict = processInput('../data/test.ini')

        model_obj = constructModel(init_dict)

        paras_obj = constructParas(init_dict, model_obj, False)

        para_objs = paras_obj.getAttr('paraObjs')
        
        for para_obj in para_objs:
            
            value = para_obj.getAttr('value')
            
            ext = paras_obj._transformToExternal(para_obj, value)
            
            int_ = paras_obj._transformToInternal(para_obj, ext)

            assert_almost_equal(value, int_)
                    
        # Cleanup.
        cleanup(resume=False)

if __name__ == '__main__': 
    
    runmodule()   
