''' Module for the management of the maximization algorithms.
'''

# standard library.
import numpy as np
from   scipy.optimize  import  fmin_bfgs, fmin_powell

# project library
from grmpy.clsMeta import metaCls
from grmpy.clsCrit import critCls


class maxCls(metaCls):
    
    def __init__(self, grmObj):
        
        self.attr = {}
        
        # Distribute class attributes.
        self.attr['grmObj']  = grmObj
        
        # Results container.
        self.attr['maxRslt'] = None
                
        # Status.
        self.isLocked = False
    
    def _derivedAttributes(self):
        ''' Construct derived attributes.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')
        
        # Criterion function.
        critFunc = critCls(grmObj)
        
        critFunc.lock()            
    
        self.attr['critFunc'] = critFunc
    
    def maximize(self):
        ''' Maximization
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Distribute class attributes.
        grmObj     = self.getAttr('grmObj')

        critFunc   = self.getAttr('critFunc')
        

        parasObj   = grmObj.getAttr('parasObj')

        requestObj = grmObj.getAttr('requestObj')        

        
        algorithm  = requestObj.attr['algorithm']
        
        maxiter    = requestObj.getAttr('maxiter')
        
        # Maximization.
        if(maxiter == 0):
            
            x        = parasObj.getValues('external', 'free')
            
            
            maxRslt = {}
            
            maxRslt['fun']     = _scipyWrapperFunction(x, critFunc)
            
            maxRslt['grad']    = _scipyWrapperGradient(x, critFunc)
            
            maxRslt['xopt']    = x
                        
            maxRslt['success'] = False
    
            # Message:
            maxRslt['message'] = 'Single function evaluation at starting values.'
                
        elif(algorithm == 'bfgs'):
            
            maxRslt = self._bfgs()
            
        elif(algorithm == 'powell'):
            
            maxRslt = self._powell()
        
        # Logging.
        self._logging(maxRslt)
        
        # Finishing.
        return maxRslt
        
    ''' Private Methods.
    '''
    def _logging(self, maxRslt):
        ''' Finalize logging.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        
        # Distribute information.
        fval = str(maxRslt['fun'])
        
        if(maxRslt['grad'] is not None):
        
            grad = str(np.amax(np.abs(maxRslt['grad'])))
        
        else:
            
            grad = 'None'
        
        success = str(maxRslt['success'])
        
        msg     = maxRslt['message']
        
        # Write to file.
        file_ = open('grmToolbox.grm.log', 'a')
        
        file_.write('''\n Optimization Report \n''')        
        
        file_.write('''\n      Function:   ''' + fval)

        file_.write('''\n      Gradient:   ''' + grad + '\n')

        file_.write('''\n      Success:    ''' + success)
        
        file_.write('''\n      Message:    ''' + msg + '\n\n\n\n')
        
        file_.close()   
        
    
    def _powell(self):
        ''' Method that performs the Powell maximization.
        '''
        # Antibugging.
        assert (self.getStatus() == True)

        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')

        requestObj = grmObj.getAttr('requestObj')

        parasObj   = grmObj.getAttr('parasObj')
                
        maxiter    = requestObj.getAttr('maxiter')

        critFunc   = self.getAttr('critFunc')
        
        # Staring values.
        startingValues = parasObj.getValues(version = 'external', which = 'free')
        
        rslt = fmin_powell(
                
                func        = _scipyWrapperFunction, 
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
        assert (self.getStatus() == True)

        # Distribute class attributes.
        grmObj = self.getAttr('grmObj')

        requestObj = grmObj.getAttr('requestObj')

        parasObj   = grmObj.getAttr('parasObj')
        
        maxiter    = requestObj.getAttr('maxiter')
        
        gtol       = requestObj.getAttr('gtol')
        
        epsilon    = requestObj.getAttr('epsilon')

        critFunc   = self.getAttr('critFunc')
                
        # Staring values.
        startingValues = parasObj.getValues(version = 'external', which = 'free')
        
        # Maximization.
        rslt = fmin_bfgs(
                    
                f           = _scipyWrapperFunction, 
                fprime      = _scipyWrapperGradient,
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
        
''' Private functions of the module.
'''
def _scipyWrapperGradient(x, critFunc):
    ''' Wrapper for the gradient calculation.
    '''
    # Antibugging.
    assert (isinstance(x, np.ndarray))
    assert (np.all(np.isfinite(x)))
    assert (x.dtype == 'float')
    assert (x.ndim == 1)
    
    assert (isinstance(critFunc, critCls))
    assert (critFunc.getStatus() == True)

    # Evaluate gradient.
    grad = critFunc.evaluate(x, 'gradient')
    
    # Check quality.
    assert (isinstance(grad, np.ndarray))
    assert (np.all(np.isfinite(grad)))
    assert (grad.dtype == 'float')
        
    return grad
    
def _scipyWrapperFunction(x, critFunc):
    ''' Wrapper for most scipy maximization algorithms.
    '''
    # Antibugging.
    assert (isinstance(x, np.ndarray))
    assert (np.all(np.isfinite(x)))
    assert (x.dtype == 'float')
    assert (x.ndim == 1)
    
    assert (isinstance(critFunc, critCls))
    assert (critFunc.getStatus() == True)
    
    # Evaluate likelihood.
    likl = critFunc.evaluate(x, 'function')
    
    # Quality checks.
    assert (isinstance(likl, float))    
    assert (np.isfinite(likl))
    
    #Finishing.        
    return likl