#!/usr/bin/env python
"""  Module that contains testing of complete estimation runs. The purpose
    is to detect immediately if there are any changes in the estimation 
    output. This allows to check if the change was intended to occur. 
    The success of the tests depends heavily on the software versions and
    system architecture. They are not expected to success on other machines 
    other than @zeus.

"""
# standard library
try:
   import cPickle as pkl
except:
   import pickle as pkl

import socket
import sys
import os

from nose.core import *
from nose.tools import *

# Pythonpath
sys.path.insert(0, os.environ['GRMPY'])

# project library
import grmpy.public as grmpy

# virtual environment
if not hasattr(sys, 'real_prefix'):
   raise AssertionError('Please use a virtual environment for testing')


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
        
        grmpy.estimate(init_file, resume=False, useSimulation=False)

        # Assessment of results
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))
        
        max_rslt = rslt_dict['maxRslt']
        
        # Assertions
        assert_almost_equal(max_rslt['fun'], 1.4838776375095368)

        # Cleanup
        grmpy.cleanup(resume=False)

    @staticmethod
    def test_est_run_two():
        """ Basic estimation run, Two.
        """
        # Run command.
        init_file = 'dat/testInit_B.ini'
        
        grmpy.estimate(init_file, resume=False, useSimulation=False)

        # Assessment of results.
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))

        # Assertions.
        assert_almost_equal(rslt_dict['maxRslt']['fun'], 1.6569860751490129)

        # Cleanup.
        grmpy.cleanup(resume=False)

    @staticmethod
    def test_est_run_three():
        """ Basic estimation run, Three.
        """
        # Run command
        init_file = 'dat/testInit_C.ini'
        
        grmpy.estimate(init_file, resume = False, useSimulation = False)

        # Assessment of results
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))

        # This test only succeeds on our testing server. Otherwise, slight
        # differences in the Python version result in a failure. It only
        # checked to work with Python 2.7.6.
        if socket.gethostname() != 'zeus':
            return True

        # Assertions
        assert_almost_equal(rslt_dict['maxRslt']['fun'], 1.628181660180656)

        assert_almost_equal(rslt_dict['bmteExPost']['estimate'][50],
        -0.10666320040952278)
        assert_almost_equal(rslt_dict['bmteExPost']['confi']['upper'][50],
        -0.079594731462436979gi)
        assert_almost_equal(rslt_dict['bmteExPost']['confi']['lower'][50],
        -0.1345556517240841)
   
        assert_almost_equal(rslt_dict['smteExAnte']['estimate'][50],
        -0.13443516496559108)
        assert_almost_equal(rslt_dict['smteExAnte']['confi']['upper'][50],
        -0.10158353097568325)
        assert_almost_equal(rslt_dict['smteExAnte']['confi']['lower'][50],
        -0.16744058087070901)
   
        #Assert relationship between parameters
        for i in range(99):
            
            cmteExAnte = rslt_dict['cmteExAnte']['estimate'][i]
            smteExAnte = rslt_dict['smteExAnte']['estimate'][i]
            bmteExPost = rslt_dict['bmteExPost']['estimate'][i]
            
            assert_almost_equal(smteExAnte, bmteExPost - cmteExAnte)
   
        # Cleanup.
        grmpy.cleanup(resume=False)

    @staticmethod
    def test_est_run_four():
        """ Basic estimation run, Four.
        """
        # Run command
        initFile = 'dat/testInit_D.ini'

        grmpy.estimate(initFile, resume=False, useSimulation=False)

        # Assessment of results.
        rslt_dict = pkl.load(open('rslt.grmpy.pkl', 'rb'))

        # This test only succeeds only when using Python 2.x.x.
        if sys.version[0] != 2:
            return True

        # Assertions.
        assert_almost_equal(rslt_dict['bteExPost']['average']['estimate'],
        -0.14337522136788611)
        assert_almost_equal(rslt_dict['bteExPost']['treated']['estimate'],
        0.065536369981413087)
        assert_almost_equal(rslt_dict['bteExPost']['untreated']['estimate'],
        -0.30752004314233555)

        assert_almost_equal(rslt_dict['bteExAnte']['average']['estimate'],
        -0.14337522136788611)
        assert_almost_equal(rslt_dict['bteExAnte']['treated']['estimate'],
        0.065536369981413087)
        assert_almost_equal(rslt_dict['bteExAnte']['untreated']['estimate'],
        -0.30752004314233555)

        assert_almost_equal(rslt_dict['cte']['average']['estimate'],
        0.078672710078639602)
        assert_almost_equal(rslt_dict['cte']['treated']['estimate'],
        -0.72163877018841327)
        assert_almost_equal(rslt_dict['cte']['untreated']['estimate'],
        0.70748887314560982)

        assert_almost_equal(rslt_dict['ste']['average']['estimate'],
        -0.22204793144652576)
        assert_almost_equal(rslt_dict['ste']['treated']['estimate'],
        0.78717514016982626)
        assert_almost_equal(rslt_dict['ste']['untreated']['estimate'],
        -1.0150089162879454)

        # Cleanup.
        grmpy.cleanup(resume=False)

if __name__ == '__main__': 
    
    runmodule()   
