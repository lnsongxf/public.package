#!/usr/bin/env python
"""  Module that contains testing of complete estimation runs. The purpose
    is to detect immediately if there are any changes in the estimation 
    output. This allows to check if the change was intended to occur. 
    The success of the tests depends heavily on the software versions and
    system architecture. They are not expected to success on other machines 
    other than @zeus.

"""
# standard library
import pickle as pkl

import socket
import glob
import sys
import os

from nose.core import runmodule
from nose.tools import assert_almost_equal

# Pythonpath
sys.path.insert(0, os.environ['GRMPY'])

# project library
import grmpy

# virtual environment
if not hasattr(sys, 'real_prefix'):
   raise AssertionError('Please use a virtual environment for testing')


'''Auxiliary functions
'''
def cleanup():
    """ Cleanup from previous estimation run.
    """
    # Construct files list
    file_list = glob.glob('*.grmpy.*')

    # Remove information from simulated data
    for file_ in ['*.infos.grmpy.out']:
        try:
            file_list.remove(glob.glob(file_)[0])
        except IndexError:
            pass

    # Cleanup
    for file_ in file_list:

        if 'ini' in file_:
            continue

        os.remove(file_)


''' Test class '''

class TestEstimationRuns(object):
    """ Testing full estimation runs for a variety of data specifications.
    """
    @staticmethod
    def test_est_run_one():
        """ Basic estimation run, One.
        """
        # Run command
        init_file = 'dat/testInit_A.ini'
        
        grmpy.estimate(init_file, resume=False, use_simulation=False)

        # Assessment of results
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))
        
        max_rslt = rslt_dict['max_rslt']
        
        # Assertions
        assert_almost_equal(max_rslt['fun'], 1.4838776375095368)

        # Cleanup
        cleanup()

    @staticmethod
    def test_est_run_two():
        """ Basic estimation run, Two.
        """
        # Run command.
        init_file = 'dat/testInit_B.ini'
        
        grmpy.estimate(init_file, resume=False, use_simulation=False)

        # Assessment of results.
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))

        # Assertions.
        assert_almost_equal(rslt_dict['max_rslt']['fun'], 1.6569860751490129)

        # Cleanup.
        cleanup()

    @staticmethod
    def test_est_run_three():
        """ Basic estimation run, Three.
        """
        # Run command
        init_file = 'dat/testInit_C.ini'
        
        grmpy.estimate(init_file, resume=False, use_simulation=False)

        # Assessment of results
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))

        # This test only succeeds on our testing server. 
        if socket.gethostname() != 'zeus':
            return True

        # Assertions
        assert_almost_equal(rslt_dict['max_rslt']['fun'], 1.628181660180656)

        assert_almost_equal(rslt_dict['bmteExPost']['estimate'][50],
        -0.10666320040952278)
        assert_almost_equal(rslt_dict['bmteExPost']['confi']['upper'][50],
        -0.079594731462436979)
        assert_almost_equal(rslt_dict['bmteExPost']['confi']['lower'][50],
        -0.1345556517240841)
   
        assert_almost_equal(rslt_dict['smteExAnte']['estimate'][50],
        -0.13443516496559108)
        assert_almost_equal(rslt_dict['smteExAnte']['confi']['upper'][50],
        -0.10380872727729094)
        assert_almost_equal(rslt_dict['smteExAnte']['confi']['lower'][50],
        -0.16786079416740679)
   
        # Assert relationship between parameters
        for i in range(99):
            
            cmteExAnte = rslt_dict['cmteExAnte']['estimate'][i]
            smteExAnte = rslt_dict['smteExAnte']['estimate'][i]
            bmteExPost = rslt_dict['bmteExPost']['estimate'][i]
            
            assert_almost_equal(smteExAnte, bmteExPost - cmteExAnte)
   
        # Cleanup.
        cleanup()

if __name__ == '__main__': 
    
    runmodule()   
