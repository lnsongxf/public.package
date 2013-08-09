#!/usr/bin/env python
''' Module that contains testing of complete estimation runs. The purpose 
    is to detect immediately if there are any changes in the estimation 
    output. This allows to check if the change was intended to occur.

'''
# standard library
import os
import sys

import  pickle      as pkl
import  numpy       as np

from    nose.core   import *
from    nose.tools  import *

# set working directory
dir_ = os.path.abspath(os.path.split(sys.argv[0])[0])
os.chdir(dir_)

# edit system path
sys.path.insert(0, dir_.replace('/tests', ''))

# project library
import grmToolbox

''' Auxiliary functions.

'''
class testEstimationRuns(object):
    ''' Testing full estimation runs for a variety of data specifications. All 
        runs are repeated with a different processor counts to ensure that
        the results are unaffected by the number of processors employed.
    
    '''
    def testEstRunOne(self):
        ''' Basic estimation run.
        '''
        
        Ypos = 0; Dpos = 1
        
        Bpos = [2, 3]; Mpos = [4]
        Cpos = [5]
        
        info    =  np.array([False, True])
        
        dataset = np.genfromtxt('../dat/testDataOne.dat')
        
        ''' Construct user request.
        '''
        userRequest = grmToolbox.userRequest()
        
        userRequest.setAttr('data', dataset)
        
        userRequest.setAttr('Ypos', Ypos)
        
        userRequest.setAttr('Dpos', Dpos)
        
        userRequest.setAttr('Bpos', Bpos)
        
        userRequest.setAttr('Mpos', Mpos)
        
        userRequest.setAttr('Cpos', Cpos)
        
        userRequest.setAttr('info', info)
        
        userRequest.setAttr('maxiter', 1)

        userRequest.setAttr('numDraws', 1000)
        
        userRequest.setAttr('isDebug', True)
                
        userRequest.lock()
        
        # Maximization routine.
        rslt = grmToolbox.maximize(userRequest)
        
        # Checks.
        assert_equal(rslt.getAttr('fun'), 1.8316796198013277)

    def testEstRunTwo(self):
        ''' Basic estimation run.
        '''
        
        Ypos = 0; Dpos = 1
        
        Bpos = None; Mpos = [2, 3, 4]
        Cpos = [5]
        
        info    =  None
        
        dataset = np.genfromtxt('../dat/testDataOne.dat')
        
        ''' Construct user request.
        '''
        userRequest = grmToolbox.userRequest()
        
        userRequest.setAttr('data', dataset)
        
        userRequest.setAttr('Ypos', Ypos)
        
        userRequest.setAttr('Dpos', Dpos)
        
        userRequest.setAttr('Bpos', Bpos)
        
        userRequest.setAttr('Mpos', Mpos)
        
        userRequest.setAttr('Cpos', Cpos)
        
        userRequest.setAttr('info', info)
        
        userRequest.setAttr('maxiter', 1)

        userRequest.setAttr('numDraws', 1000)

        userRequest.setAttr('isDebug', True)

        userRequest.lock()
        
        # Maximization routine.
        rslt = grmToolbox.maximize(userRequest)
            
        # Checks.
        assert_equal(rslt.getAttr('fun'), 1.7879213046750069)  
        
        assert_almost_equal(rslt.getAttr('bmteExPost')['estimate'][50],        0.025258916692701295)
        assert_almost_equal(rslt.getAttr('bmteExPost')['confi']['upper'][50],  0.053722025588227758)   
        assert_almost_equal(rslt.getAttr('bmteExPost')['confi']['lower'][50], -0.002952890517264570)   
        
        
        os.unlink('rslt.pkl')

    def testEstRunThree(self):
        ''' Basic estimation run.
        '''
        
        Ypos = 0; Dpos = 1
        
        Bpos = [2, 3]
        info = np.array([True, True])
        Mpos = [4]
        Cpos = [5]
     
        
        dataset = np.genfromtxt('../dat/testDataOne.dat')
        
        ''' Construct user request.
        '''
        userRequest = grmToolbox.userRequest()
        
        userRequest.setAttr('data', dataset)
        
        userRequest.setAttr('Ypos', Ypos)
        
        userRequest.setAttr('Dpos', Dpos)
        
        userRequest.setAttr('Bpos', Bpos)
        
        userRequest.setAttr('Mpos', Mpos)
        
        userRequest.setAttr('Cpos', Cpos)
        
        userRequest.setAttr('info', info)
        
        userRequest.setAttr('maxiter', 1)

        userRequest.setAttr('numDraws', 1000)

        userRequest.setAttr('isDebug', True)

        userRequest.lock()
        
        # Maximization routine.
        rslt = grmToolbox.maximize(userRequest)
            
        # Checks.
        assert_equal(rslt.getAttr('fun'), 1.8083935923475638)  
        
        assert_almost_equal(rslt.getAttr('bmteExPost')['estimate'][50],        0.024626870566593968)
        assert_almost_equal(rslt.getAttr('bmteExPost')['confi']['upper'][50],  0.050170067455248921)   
        assert_almost_equal(rslt.getAttr('bmteExPost')['confi']['lower'][50], -0.001505978635197130)   
   
        assert_almost_equal(rslt.getAttr('smteExAnte')['estimate'][50],       -0.048225377608784627)
        assert_almost_equal(rslt.getAttr('smteExAnte')['confi']['upper'][50], -0.014838924923621277)   
        assert_almost_equal(rslt.getAttr('smteExAnte')['confi']['lower'][50], -0.08273402196467225)   
   
        #Assert relationship between parameters. 
        for i in range(99):
            
            cmteExAnte = rslt.getAttr('cmteExAnte')['estimate'][i]
            smteExAnte = rslt.getAttr('smteExAnte')['estimate'][i]
            bmteExPost = rslt.getAttr('bmteExPost')['estimate'][i]
            
            assert_almost_equal(smteExAnte, bmteExPost - cmteExAnte)
   
        os.unlink('rslt.pkl')
   
    def testEstRunFour(self):
        ''' Basic estimation run.
        '''
        
        Ypos = 0; Dpos = 1
        
        Bpos = [2, 3]
        info = np.array([True, True])
        Mpos = [4]
        Cpos = [5]
     
        
        dataset = np.genfromtxt('../dat/testDataOne.dat')
        
        ''' Construct user request.
        '''
        userRequest = grmToolbox.userRequest()
        
        userRequest.setAttr('data', dataset)
        
        userRequest.setAttr('Ypos', Ypos)
        
        userRequest.setAttr('Dpos', Dpos)
        
        userRequest.setAttr('Bpos', Bpos)
        
        userRequest.setAttr('Mpos', Mpos)
        
        userRequest.setAttr('Cpos', Cpos)
        
        userRequest.setAttr('info', info)
        
        userRequest.setAttr('maxiter', 1)

        userRequest.setAttr('numDraws', 100)

        userRequest.setAttr('isDebug', True)

        userRequest.setAttr('numSims', 100)
        
        userRequest.setAttr('withAverageEffects', True)
                
        userRequest.lock()
        
        # Maximization routine.
        rslt = grmToolbox.maximize(userRequest)
            
        # Checks.
        assert_almost_equal(rslt.getAttr('bteExPost')['average']['estimate'], 0.064544458629204854)
        assert_almost_equal(rslt.getAttr('bteExPost')['treated']['estimate'], 0.13096680159900997)
        assert_almost_equal(rslt.getAttr('bteExPost')['untreated']['estimate'], 0.00072691342292147674)

        assert_almost_equal(rslt.getAttr('bteExAnte')['average']['estimate'], 0.064544458629204854)
        assert_almost_equal(rslt.getAttr('bteExAnte')['treated']['estimate'], 0.13096680159900997)
        assert_almost_equal(rslt.getAttr('bteExAnte')['untreated']['estimate'], 0.00072691342292147674)

        assert_almost_equal(rslt.getAttr('cte')['average']['estimate'],   0.319393069747761)
        assert_almost_equal(rslt.getAttr('cte')['treated']['estimate'],  -1.948504047426586)
        assert_almost_equal(rslt.getAttr('cte')['untreated']['estimate'], 2.498353045072135)

        assert_almost_equal(rslt.getAttr('ste')['average']['estimate'],  -0.25484861111855711)
        assert_almost_equal(rslt.getAttr('ste')['treated']['estimate'],   2.0794708490255962)
        assert_almost_equal(rslt.getAttr('ste')['untreated']['estimate'],-2.4976261316492132)

        os.unlink('rslt.pkl')
        
if __name__ == '__main__':
    
    runmodule()   