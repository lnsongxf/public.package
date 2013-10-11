#!/usr/bin/env python
''' Module that contains testing of complete estimation runs. The purpose 
    is to detect immediately if there are any changes in the estimation 
    output. This allows to check if the change was intended to occur.

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

''' Test class.
'''
class testEstimationRuns(object):
    ''' Testing full estimation runs for a variety of data specifications.
    '''
    def testEstRunOne(self):
        ''' Basic estimation run.
        '''
        
        # Run command.
        initFile = '../dat/testCase_First.ini'
        
        os.system('../bin/grmToolbox-estimation -init ' + initFile)
        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        maxRslt = rsltObj.getAttr('maxRslt')
        
        # Assertions.
        assert_equal(maxRslt['fun'], 1.643038068973831)

        # Cleanup.
        os.system('../bin/grmToolbox-cleanup')
        
    def testEstRunTwo(self):
        ''' Basic estimation run.
        '''
        
        # Run command.
        initFile = '../dat/testCase_Second.ini'
        
        os.system('../bin/grmToolbox-estimation -init ' + initFile)
        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        maxRslt = rsltObj.getAttr('maxRslt')
        
        # Assertions.
        assert_equal(maxRslt['fun'], 1.6569859824560313)  
        
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['estimate'][50],       -0.10666298513882175)
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['upper'][50], -0.07799826634461817)   
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['lower'][50], -0.13482394691373967)   

        # Cleanup.
        os.system('../bin/grmToolbox-cleanup')

    def testEstRunThree(self):
        ''' Basic estimation run.
        '''
        
        # Run command.
        initFile = '../dat/testCase_Third.ini'
        
        os.system('../bin/grmToolbox-estimation -init ' + initFile)
        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        maxRslt = rsltObj.getAttr('maxRslt')
        
        # Assertions.
        assert_equal(maxRslt['fun'], 1.6281817748415393)  
        
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['estimate'][50],       -0.10666298513882175)
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['upper'][50], -0.08054064280753470)   
        assert_almost_equal(rsltObj.getAttr('bmteExPost')['confi']['lower'][50], -0.13243970064779459)   
   
        assert_almost_equal(rsltObj.getAttr('smteExAnte')['estimate'][50],       -0.13443440314930999)
        assert_almost_equal(rsltObj.getAttr('smteExAnte')['confi']['upper'][50], -0.10006386091593558)   
        assert_almost_equal(rsltObj.getAttr('smteExAnte')['confi']['lower'][50], -0.16483455296419108)   
   
        #Assert relationship between parameters. 
        for i in range(99):
            
            cmteExAnte = rsltObj.getAttr('cmteExAnte')['estimate'][i]
            smteExAnte = rsltObj.getAttr('smteExAnte')['estimate'][i]
            bmteExPost = rsltObj.getAttr('bmteExPost')['estimate'][i]
            
            assert_almost_equal(smteExAnte, bmteExPost - cmteExAnte)
   
        # Cleanup.
        os.system('../bin/grmToolbox-cleanup')
  
    def testEstRunFour(self):
        ''' Basic estimation run.
        '''
   
        # Run command.
        initFile = '../dat/testCase_Fourth.ini'
        
        os.system('../bin/grmToolbox-estimation -init ' + initFile)
        
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
        os.system('../bin/grmToolbox-cleanup')

    def testEstRunFive(self):
        ''' Basic estimation run.
        '''
        
        # Run command.
        initFile = '../dat/testCase_Fifth.ini'
        
        os.system('../bin/grmToolbox-estimation -init ' + initFile)
        
        # Assessment of results.
        rsltObj = pkl.load(open('rsltObj.grm.pkl', 'r'))
        
        sdV     = rsltObj.getAttr('parasObj').getParameters('sd', 'V', isObj = False) 
        
        # Assertions.
        assert_almost_equal(sdV, 0.22673338826690217)
        
        # Cleanup.
        os.system('../bin/grmToolbox-cleanup')
        
if __name__ == '__main__': 
    
    runmodule()   
