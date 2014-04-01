#!/usr/bin/env python
''' Module that contains testing of complete estimation runs. The purpose 
    is to detect immediately if there are any changes in the estimation 
    output. This allows to check if the change was intended to occur. 
    The success of the tests depends heavily on the software versions and
    system architecture. They are not expected to success on other machines 
    other than @heracles.

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

from scripts.estimate import estimate

''' Test class.
'''
class testEstimationRuns(object):
    ''' Testing full estimation runs for a variety of data specifications.
    '''
    def testEstRunOne(self):
        ''' Basic estimation run, One.
        '''
        
        # Run command.
        initFile = '../dat/testInit_A.ini'
        
        estimate(initFile, restart = False, useSimulation = False)

        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        maxRslt = rsltObj.getAttr('maxRslt')
        
        # Assertions.
        assert_almost_equal(maxRslt['fun'], 1.643038068973831)

        # Cleanup.
        grmToolbox.cleanup(isRestart = False)
        
    def testEstRunTwo(self):
        ''' Basic estimation run, Two.
        '''
        
        # Run command.
        initFile = '../dat/testInit_B.ini'
        
        estimate(initFile, restart = False, useSimulation = False)

        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        maxRslt = rsltObj.getAttr('maxRslt')
        
        # Assertions.
        assert_almost_equal(maxRslt['fun'], 1.6569859824560313)  
        
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['estimate'][50],       -0.10666298513882175)
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['upper'][50], -0.07895223145741086)   
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['lower'][50], -0.13216952695996106)   

        # Cleanup.
        grmToolbox.cleanup(isRestart = False)

    def testEstRunThree(self):
        ''' Basic estimation run, Three
        '''
        
        # Run command.
        initFile = '../dat/testInit_C.ini'
        
        estimate(initFile, restart = False, useSimulation = False)

        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        maxRslt = rsltObj.getAttr('maxRslt')
        
        # Assertions.
        assert_almost_equal(maxRslt['fun'], 1.6281817748415393)  
        
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['estimate'][50],       -0.10666298513882175)
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['upper'][50], -0.07812857784657281)   
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['lower'][50], -0.13354217907346827)   
   
        assert_almost_equal(rsltObj.getAttr('smteExAnte')['estimate'][50],       -0.13443440314930999)
        assert_almost_equal(rsltObj.getAttr('smteExAnte')['confi']['upper'][50], -0.10098384999873594)   
        assert_almost_equal(rsltObj.getAttr('smteExAnte')['confi']['lower'][50], -0.16775667307112982)   
   
        #Assert relationship between parameters. 
        for i in range(99):
            
            cmteExAnte = rsltObj.getAttr('cmteExAnte')['estimate'][i]
            smteExAnte = rsltObj.getAttr('smteExAnte')['estimate'][i]
            bmteExPost = rsltObj.getAttr('bmteExPost')['estimate'][i]
            
            assert_almost_equal(smteExAnte, bmteExPost - cmteExAnte)
   
        # Cleanup.
        grmToolbox.cleanup(isRestart = False)
  
    def testEstRunFour(self):
        ''' Basic estimation run, Four.
        '''
   
        # Run command.
        initFile = '../dat/testInit_D.ini'
        
        estimate(initFile, restart = False, useSimulation = False)

        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))

        # Assertions.
        assert_almost_equal(rsltObj.getAttr('bteExPost')['average']['estimate'],    -0.14337500688760152)
        assert_almost_equal(rsltObj.getAttr('bteExPost')['treated']['estimate'],     0.06553651561690533)
        assert_almost_equal(rsltObj.getAttr('bteExPost')['untreated']['estimate'],  -0.30751977456971408)

        assert_almost_equal(rsltObj.getAttr('bteExAnte')['average']['estimate'],    -0.14337500688760152)
        assert_almost_equal(rsltObj.getAttr('bteExAnte')['treated']['estimate'],     0.06553651561690533)
        assert_almost_equal(rsltObj.getAttr('bteExAnte')['untreated']['estimate'],  -0.30751977456971408)

        assert_almost_equal(rsltObj.getAttr('cte')['average']['estimate'],    0.0786721631745564)
        assert_almost_equal(rsltObj.getAttr('cte')['treated']['estimate'],   -0.7216392651098148)
        assert_almost_equal(rsltObj.getAttr('cte')['untreated']['estimate'],  0.7074882853979908)

        assert_almost_equal(rsltObj.getAttr('ste')['average']['estimate'],   -0.2220471700621577)
        assert_almost_equal(rsltObj.getAttr('ste')['treated']['estimate'],    0.7871757807267197)
        assert_almost_equal(rsltObj.getAttr('ste')['untreated']['estimate'], -1.0150080599677049)

        # Cleanup.
        grmToolbox.cleanup(isRestart = False)
        
if __name__ == '__main__': 
    
    runmodule()   
