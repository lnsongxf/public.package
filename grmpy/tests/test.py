#!/usr/bin/env python
""" Module for unit tests related to the parameter management and updating.
"""
# standard library
import shutil
import glob
import sys
import os

# module variables
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.getcwd()

# testing library
from nose.core import runmodule
from nose.tools import *

# PYTHONPATH
dir_ = FILE_PATH.replace('/tests', '')
sys.path.insert(0, dir_)

# project library
from grmpy.tools.user import *


''' Test class.
'''
class TestParasCls(object):
    """ Test class.
    """
    @staticmethod
    def setup():
        os.chdir(FILE_PATH)

    @staticmethod
    def teardown():
        os.chdir(TEST_PATH)

    @staticmethod
    def cleanup():
        """ Cleanup after test battery.
        """
        files = []

        files = files + glob.glob('*.grmpy.*')

        files = files + glob.glob('*.ini')

        files = files + glob.glob('*.pkl')

        files = files + glob.glob('*.txt')

        files = files + glob.glob('*.dat')

        for file_ in files:

            if 'logging' in file_:
                continue

            try:

                os.remove(file_)

            except OSError:

                pass

            try:

                shutil.rmtree(file_)

            except OSError:

                pass

    def test_a(self):
        """ Test parameter transformations.
        """
        init_dict = process_input('../data/test.ini')

        model_obj = construct_model(init_dict)

        paras_obj = construct_paras(init_dict, model_obj, False)

        para_objs = paras_obj.get_attr('para_objs')
        
        for para_obj in para_objs:
            
            value = para_obj.get_attr('value')
            
            ext = paras_obj._transform_to_external(para_obj, value)
            
            int_ = paras_obj._transform_to_internal(para_obj, ext)

            assert_almost_equal(value, int_)

        # Cleanup.
        self.cleanup()



if __name__ == '__main__':
    
    runmodule()   
