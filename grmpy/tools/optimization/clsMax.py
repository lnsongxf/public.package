''' Module for the management of the maximization algorithms.
'''

# standard library.
import numpy as np
from   scipy.optimize  import  fmin_bfgs, fmin_powell

# project library
from grmpy.clsMeta import metaCls
from grmpy.tools.optimization.clsCrit import critCls
from grmpy.clsModel import modelCls
from grmpy.clsParas import parasCls

from grmpy.tools.optimization.wrappers import *

class maxCls(metaCls):
    
    def __init__(self, model_obj, paras_obj):

        # Antibugging.
        assert (isinstance(model_obj, modelCls))
        assert (isinstance(paras_obj, parasCls))

        assert (model_obj.get_status() == True)
        assert (paras_obj.get_status() == True)

        self.attr = dict()

        self.attr['model_obj'] = model_obj

        self.attr['paras_obj'] = paras_obj

        # Results container.
        self.attr['maxRslt'] = None
                
        # Status.
        self.isLocked = False
    
    def _derivedAttributes(self):
        ''' Construct derived attributes.
        '''
        # Antibugging.
        assert (self.get_status() == True)
        
        # Distribute class attributes.
        model_obj = self.getAttr('model_obj')
        paras_obj = self.getAttr('paras_obj')

        # Criterion function.
        critFunc = critCls(model_obj, paras_obj)
        
        critFunc.lock()            
    
        self.attr['critFunc'] = critFunc
    
    def maximize(self):
        ''' Maximization
        '''
        # Antibugging.
        assert (self.get_status() == True)
        
        # Distribute class attributes.
        critFunc   = self.getAttr('critFunc')

        paras_obj   = self.getAttr('paras_obj')

        model_obj = self.getAttr('model_obj')

        algorithm  = model_obj.getAttr('algorithm')
        
        maxiter    = model_obj.getAttr('maxiter')
        
        # Maximization.
        if(maxiter == 0):
            
            x        = paras_obj.getValues('external', 'free')
            
            
            maxRslt = {}
            
            maxRslt['fun']     = scipy_wrapper_function(x, critFunc)
            
            maxRslt['grad']    = scipy_wrapper_gradient(x, critFunc)
            
            maxRslt['xopt']    = x
                        
            maxRslt['success'] = False
    
            # Message:
            maxRslt['message'] = 'Single function evaluation at starting values.'
                
        elif(algorithm == 'bfgs'):
            
            maxRslt = self._bfgs()
            
        elif(algorithm == 'powell'):
            
            maxRslt = self._powell()

        # Finishing.
        return maxRslt
        
    ''' Private Methods.
    '''
    def _powell(self):
        ''' Method that performs the Powell maximization.
        '''
        # Antibugging.
        assert (self.get_status() == True)

        # Distribute class attributes.
        model_obj = self.getAttr('model_obj')

        paras_obj   = self.getAttr('paras_obj')
                
        maxiter    = model_obj.getAttr('maxiter')

        critFunc   = self.getAttr('critFunc')
        
        # Staring values.
        startingValues = paras_obj.getValues(version = 'external', which = 'free')
        
        rslt = fmin_powell(
                
                func        = scipy_wrapper_function,
                x0          = startingValues, 
                args        = (critFunc, ), 
                xtol        = 0.0000000001,
                ftol        = 0.0000000001,
                maxiter     = maxiter,
                maxfun      = None,
                full_output = True,
                disp        = 1,
                callback    = None
                              
            )
        
        # Prepare result dictionary.
        maxRslt = {}
        
        maxRslt['xopt']    = rslt[0]
        maxRslt['fun']     = rslt[1]
        maxRslt['grad']    = None
        maxRslt['success'] = (rslt[5] == 0)
        
        # Message.
        maxRslt['message']  = rslt[5]
        
        if(maxRslt['message'] == 1):
            
            maxRslt['message'] = 'Maximum number of function evaluations.'
            
        if(maxRslt['message'] == 0):
            
            maxRslt['message'] = 'None'
            
        if(maxRslt['message'] == 2):
            
            maxRslt['message'] = 'Maximum number of iterations.'       
            
        # Finishing.
        return maxRslt
        
    def _bfgs(self):
        ''' Method that performs a BFGS maximization.
        '''
        # Antibugging.
        assert (self.get_status() == True)

        # Distribute class attributes.
        model_obj = self.getAttr('model_obj')

        paras_obj   = self.getAttr('paras_obj')
        
        maxiter    = model_obj.getAttr('maxiter')
        
        gtol       = model_obj.getAttr('gtol')
        
        epsilon    = model_obj.getAttr('epsilon')

        critFunc   = self.getAttr('critFunc')
                
        # Staring values.
        startingValues = paras_obj.getValues(version = 'external', which = 'free')
        
        # Maximization.
        rslt = fmin_bfgs(
                    
                f           = scipy_wrapper_function,
                fprime      = scipy_wrapper_gradient,
                x0          = startingValues,
                args        = (critFunc, ), 
                gtol        = gtol,
                epsilon     = epsilon,
                maxiter     = maxiter, 
                full_output = True,
                disp        = 1,
                retall      = 0,
                callback    = None
                
            )
        
        # Prepare result dictionary.
        maxRslt = {}
       
        maxRslt['xopt']    = rslt[0]
        maxRslt['fun']     = rslt[1]
        maxRslt['grad']    = rslt[2]
        maxRslt['covMat']  = rslt[3]
        maxRslt['success'] = (rslt[6] == 0)

        # Message:
        maxRslt['message'] = rslt[6]
        
        if(maxRslt['message'] == 1):
            
            maxRslt['message'] = 'Maximum number of function evaluations.'
            
        if(maxRslt['message'] == 0):
            
            maxRslt['message'] = 'None'
            
        if(maxRslt['message'] == 2):
            
            maxRslt['message'] = 'Gradient and/or function calls not changing.'
            
        # Finishing.
        return maxRslt
